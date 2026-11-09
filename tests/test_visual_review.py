"""Regressions for raw-colour crops, input geometry, and uncertain VLM output."""
import base64
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cognition import vlm_api
from cognition.visual_input import clip_source_box, focus_views
from review.disposition import gray_decision_evidence
from review.disposition import InspectionPolicy


def preprocess():
    resize = type("Resize", (), {"size": 224, "max_size": None})()
    centre = type("CenterCrop", (), {"size": (224, 224)})()
    return SimpleNamespace(transforms=[resize, centre])


class VisualReviewTests(unittest.TestCase):
    def test_rectangular_geometry_and_crops_preserve_source_pixels(self):
        pixels = np.random.default_rng(5).integers(0, 256, (200, 600, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        np.testing.assert_allclose(clip_source_box(image, preprocess()), (200, 0, 400, 200))
        scores = np.zeros((14, 14)); scores[3, 12] = 1; scores[11, 1] = .8
        crops, info = focus_views(image, scores, preprocess())
        self.assertEqual(len(crops), 2)
        for crop, (x0, y0, x1, y1) in zip(crops, info["boxes"]):
            self.assertGreaterEqual(x0, 150)
            self.assertLessEqual(x1, 450)
            np.testing.assert_array_equal(np.asarray(crop), pixels[y0:y1, x0:x1])
        self.assertNotEqual(info["boxes"][0], info["boxes"][1])

    def test_no_fabricated_focus_when_location_is_unavailable(self):
        image = Image.new("RGB", (256, 256))
        for scores in (None, np.ones((14, 14)), np.full((14, 14), np.nan), np.ones(14)):
            self.assertEqual(focus_views(image, scores, preprocess())[0], [])
        scores = np.eye(14)
        self.assertEqual(focus_views(image, scores, None)[1]["reason"], "unsupported_preprocess")

    def test_request_sends_full_image_and_original_colour_crops_as_png(self):
        image = Image.new("RGB", (150, 180), (10, 80, 170))
        crop = image.crop((10, 20, 70, 80))
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "original.png"); image.save(path)
            with patch.object(vlm_api.requests, "post", return_value=response) as post:
                vlm_api._call_openai_compat("https://example.invalid", "fixture", "fixture", path,
                    "inspect independently", focus_images=[crop], system_prompt=vlm_api.VISUAL_SYSTEM_PROMPT)
            payload = post.call_args.kwargs["json"]
            views = [x["image_url"]["url"] for x in payload["messages"][1]["content"] if x["type"] == "image_url"]
            self.assertEqual(len(views), 2)
            for data, expected in zip(views, (image, crop)):
                self.assertTrue(data.startswith("data:image/png;base64,"))
                actual = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1])))
                np.testing.assert_array_equal(actual, expected)

    def test_conflicting_or_missing_visual_evidence_cannot_auto_archive(self):
        clear = {"ok": True, "review_version": "focus-v2", "visual_assessment": "visible_defect",
                 "image_adequacy": "adequate", "conclusion": "不合格", "defect_type": "划痕",
                 "defect_location": "原图左下方", "defect_desc": "不规则沟槽横穿加工纹理",
                 "citation_check": {"ok": True, "has_citations": True}}
        self.assertEqual(gray_decision_evidence(clear, .6, .3, .7)["proposal"], "defect")
        for update in ({"visual_assessment": "uncertain"}, {"image_adequacy": "inadequate"},
                       {"defect_location": ""}, {"conclusion": "合格"}):
            self.assertEqual(gray_decision_evidence({**clear, **update}, .6, .3, .7)["proposal"], "abstain")
        # Prompt/schema changes cannot bypass the existing position/qualification gate.
        self.assertEqual(gray_decision_evidence(clear, .35, .3, .7)["proposal"], "abstain")

    def test_inconsistent_model_json_is_forced_to_uncertain(self):
        cases = [
            {"visual_assessment": "visible_defect", "image_adequacy": "adequate", "conclusion": "合格"},
            {"visual_assessment": "no_visible_defect", "image_adequacy": "inadequate", "conclusion": "合格"},
            {"conclusion": "合格"},
        ]
        with patch.object(vlm_api, "DASHSCOPE_API_KEY", "fixture"):
            for case in cases:
                raw = json.dumps({"defect_type": "正常", "defect_desc": "fixture", **case})
                with patch.object(vlm_api, "_call_openai_compat", return_value=raw):
                    result = vlm_api.analyze("unused", "fixture", provider="dashscope", visual_review=True)
                    self.assertTrue(result["ok"], result)
                    self.assertEqual(result["conclusion"], "需复核")
                    self.assertEqual(result["visual_assessment"], "uncertain")

    def test_unvalidated_focus_version_keeps_explicit_observations_in_manual_queue(self):
        policy = InspectionPolicy.__new__(InspectionPolicy)
        policy.settings = Mock(return_value={"mode": "assisted", "sample_rate": .1, "revision": 1, "holds": {}})
        policy.profile = Mock(return_value={"model_key": "fixture", "auto_low": .2, "auto_high": .8,
                                           "allow_normal": True, "allow_isolate": True})
        policy.scope_check = Mock(return_value={"ok": True})
        record = {"score": .6, "quality_check": {"ok": True}, "gate": {"route": "vlm_review"},
                  "cognition": {"ok": True, "review_version": "focus-v2", "conclusion": "不合格",
                    "visual_assessment": "visible_defect", "image_adequacy": "adequate", "defect_type": "划痕",
                    "defect_desc": "不规则沟槽", "defect_location": "左上", "citation_check": {"ok": True, "has_citations": True}}}
        action = policy.decide(record, {"key": "screw", "threshold": .3, "t_high": .7})
        self.assertEqual(action["reason_code"], "unvalidated_visual_review")
        self.assertTrue(action["requires_review"])
        self.assertEqual(action["gray_assessment"]["proposal"], "defect")
        self.assertIsNone(action["auto_result"])

    def test_unverified_standard_advice_and_unlisted_defect_names_are_downgraded(self):
        response = {"visual_assessment": "visible_defect", "image_adequacy": "adequate", "conclusion": "不合格",
                    "defect_type": "夹杂", "defect_desc": "黑色斑点", "defect_location": "左侧",
                    "suggestion": "按照GB/T 5779.2的表面夹杂物条款报废"}
        with patch.object(vlm_api, "DASHSCOPE_API_KEY", "fixture"), \
             patch.object(vlm_api, "_call_openai_compat", return_value=json.dumps(response)):
            result = vlm_api.analyze("unused", "fixture", provider="dashscope", visual_review=True)
        self.assertEqual(result["conclusion"], "需复核")
        self.assertEqual(result["defect_type"], "其他")
        self.assertTrue(result["unverified_standard_claim"])
        self.assertNotIn("GB/T", result["suggestion"])
        self.assertIn("GB/T", result["raw"])

    def test_normal_website_entry_passes_focus_views_and_archives_geometry(self):
        from fastapi.testclient import TestClient
        from web import server
        from interface import report
        from review.workflow import ReviewWorkflow
        image = Image.fromarray(np.random.default_rng(12).integers(0, 256, (256, 256, 3), dtype=np.uint8))
        content = io.BytesIO(); image.save(content, format="PNG")
        engine = SimpleNamespace(preprocess=preprocess(), detect=lambda _image: {
            "score": .5, "heatmap": image, "anomaly_map": np.eye(14), "classification": {}})
        config = {"key": "metal_nut", "name": "fixture", "threshold": .3, "t_high": .7}
        with tempfile.TemporaryDirectory() as tmp:
            workflow = ReviewWorkflow(Path(tmp), Path(tmp, "results"))
            try:
                with patch.dict(os.environ, {"VLM_FOCUS_REVIEW": "1"}), \
                     patch.object(server, "RESULTS_DIR", str(Path(tmp, "results"))), \
                     patch.object(server, "get_review_workflow", return_value=workflow), \
                     patch.object(server, "get_engine", return_value=(engine, config)), \
                     patch.object(server, "get_clf", return_value=SimpleNamespace(available=False)), \
                     patch.object(server, "get_sem_kb", return_value=SimpleNamespace(retrieve=lambda *a, **k: [])), \
                     patch.object(report, "HISTORY_DIR", str(Path(tmp, "results/history"))), \
                     patch.object(server.vlm_api, "analyze", return_value={"ok": True, "conclusion": "需复核"}) as analyze:
                    result = TestClient(server.app).post("/api/detect", files={
                        "file": ("fixture.png", content.getvalue(), "image/png")}, data={"product": "metal_nut"}).json()
                    self.assertTrue(result["ok"], result)
                    self.assertTrue(analyze.call_args.kwargs["visual_review"])
                    self.assertEqual(len(analyze.call_args.kwargs["focus_images"]), 2)
                    self.assertEqual(len(result["cognition"]["visual_input"]["boxes"]), 2)
                    saved = list(Path(tmp, "results/history").glob("*/*.json"))
                    self.assertEqual(len(saved), 1)
                    record = json.loads(saved[0].read_text(encoding="utf-8"))
                    self.assertEqual(record["cognition"]["visual_input"], result["cognition"]["visual_input"])
            finally:
                workflow.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
