"""Regression tests for the observed probability failure and memory pollution."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perception.defect_classifier import DefectClassifier
from perception.winclip_patch import WinCLIPPatch
from web import server


class PerceptionSafetyTests(unittest.TestCase):
    def test_portable_probabilities_match_sklearn_binary_and_multiclass(self):
        rng = np.random.default_rng(17)
        for classes in (2, 3):
            with self.subTest(classes=classes), tempfile.TemporaryDirectory() as temp:
                x = rng.normal(size=(90, 6))
                y = np.arange(len(x)) % classes
                model = Pipeline([("sc", StandardScaler()),
                                  ("clf", LogisticRegression(max_iter=1000))]).fit(x, y)
                expected = model.predict_proba(x[:8])
                # Reproduce the missing attribute in the saved newer-version files.
                if hasattr(model.named_steps["clf"], "multi_class"):
                    delattr(model.named_steps["clf"], "multi_class")
                path = Path(temp, "classifier.joblib")
                joblib.dump({"clf": model}, path)
                portable = DefectClassifier(str(path))
                actual = [list(portable.predict_features(row)["probs"].values()) for row in x[:8]]
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
                with self.assertRaises(ValueError):
                    portable.predict_features(np.zeros(5))

    def test_unconfirmed_learning_is_rejected_before_feature_extraction(self):
        detector = WinCLIPPatch.__new__(WinCLIPPatch)
        with self.assertRaisesRegex(ValueError, "正常记忆库"):
            detector.learn("nonexistent.png")

    def test_confirmed_defect_enters_candidate_pool_without_touching_bank(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            day = root / "history/2026-10-22"
            day.mkdir(parents=True)
            bank = root / "bank.npy"
            np.save(bank, np.ones((196, 3), dtype=np.float32))
            old_hash = hashlib.sha256(bank.read_bytes()).hexdigest()
            record = day / "120000_fixture.json"
            record.write_text(json.dumps({"session_id": "fixture", "product_key": "metal_nut",
                                          "input_png": str(root / "image.png")}), encoding="utf-8")
            with patch.object(server, "RESULTS_DIR", str(root)), \
                 patch.object(server, "get_engine", side_effect=AssertionError("must not mutate memory")):
                response = TestClient(server.app).post("/api/confirm", json={
                    "session_id": "fixture", "verdict": "defect", "defect_type": "划痕"})
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()["learned"])
            self.assertTrue(response.json()["candidate_saved"])
            self.assertEqual(hashlib.sha256(bank.read_bytes()).hexdigest(), old_hash)
            candidates = list((root / "feedback_candidates").glob("*.json"))
            self.assertEqual(len(candidates), 1)
            candidate = json.loads(candidates[0].read_text(encoding="utf-8"))
            self.assertEqual(candidate["label"]["verdict"], "defect")
            self.assertEqual(candidate["status"], "pending_offline_review")

    def test_unknown_product_never_silently_uses_metal_nut(self):
        with self.assertRaisesRegex(ValueError, "未知产品"):
            server.get_engine("not-a-product")

    def test_optimized_model_rejects_threshold_from_a_different_score_system(self):
        from types import SimpleNamespace
        artifact = SimpleNamespace(metadata={"product": "metal_nut", "threshold": .6})
        with patch.object(server, "_optimized_models", {}), \
             patch("perception.feature_backends.OptimizedPerception", return_value=artifact):
            with self.assertRaisesRegex(ValueError, "阈值不匹配"):
                server.apply_optimized_perception("metal_nut", {
                    "perception_artifact": "fixture.json", "threshold": .2345}, None, None, {})


if __name__ == "__main__":
    unittest.main()
