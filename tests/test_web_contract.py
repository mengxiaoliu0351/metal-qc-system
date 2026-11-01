"""Web response checks with fake inference; no weights, cloud calls or real history writes.

Run: python -m unittest discover -s tests -p test_web_contract.py
"""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from PIL import Image
from web import server
from interface import report


class WebContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def test_local_assets_and_product_thresholds(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertNotIn("cdn.", page.text)
        for asset in ("app.js", "style.css"):
            self.assertEqual(self.client.get("/static/" + asset).status_code, 200)
        products = self.client.get("/api/products").json()["products"]
        self.assertEqual(len(products), 3)
        for product in products:
            self.assertGreater(product["t_high"], product["threshold"])
        self.assertTrue(all(p["method"] for p in products))

    def test_high_threshold_matches_routing_rules(self):
        self.assertEqual(server.product_high_threshold({"threshold": .2, "t_high": .3}), .3)
        self.assertEqual(server.product_high_threshold({"threshold": .2,
                         "calibration": {"median": .18, "mad": .02}}), .24)
        self.assertEqual(server.product_high_threshold({"threshold": .2}), .3)
        self.assertEqual(server.product_high_threshold({"threshold": .2, "t_high": .1}), .3)

    def test_feedback_flags_review_without_training_or_changing_prediction(self):
        from review.workflow import ReviewWorkflow, write, read
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / 'results'
            image = root / 'sample.png'
            Image.new('RGB', (16, 16)).save(image)
            path = results / 'history/2026-10-25/session.json'
            write(path, {'session_id': 'session', 'product_key': 'metal_nut', 'input_png': str(image),
                         'sample_name': 'sample', 'perception_prediction': 'normal',
                         'gate': {'route': 'low_risk'}})
            workflow = ReviewWorkflow(root, results)
            try:
                with patch.object(server, 'get_review_workflow', return_value=workflow), patch.object(report, 'HISTORY_DIR', str(results / 'history')):
                    self.assertEqual(self.client.post('/api/feedback', json={'session_id': 'missing', 'label': 'down'}).status_code, 404)
                    self.assertEqual(self.client.post('/api/feedback', json={'session_id': 'session', 'label': 'bad'}).status_code, 400)
                    up = self.client.post('/api/feedback', json={'session_id': 'session', 'label': 'up'}).json()
                    self.assertFalse(up['needs_review'])
                    down = self.client.post('/api/feedback', json={'session_id': 'session', 'label': 'down', 'comment': 'possible miss'}).json()
                    self.assertTrue(down['needs_review'])
                    self.assertFalse(down['learned'])
                    self.assertFalse(down['candidate_saved'])
                    record = read(path)
                    self.assertEqual(record['perception_prediction'], 'normal')
                    self.assertNotIn('human_verdict', record)
                    self.assertTrue(Path(record['difficult_sample_path']).exists())
                    self.assertEqual(workflow.queue(status='flagged')['count'], 1)
                    self.assertEqual(workflow.approved('metal_nut'), [])
            finally:
                workflow.executor.shutdown(wait=True)

    def test_three_routes_return_consistent_ui_metadata(self):
        image = Image.new("RGB", (128, 128))
        image.putdata([(i % 256,) * 3 for i in range(128 * 128)])
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        config = {"key": "metal_nut", "name": "test product", "threshold": .2345, "t_high": .3}
        engine = Mock()
        classifier = SimpleNamespace(available=True, predict=lambda *_, **__: {
            "label": "划痕", "confidence": .9, "probs": {"划痕": .9, "正常": .1}})
        refs = [{"id": "TEST-REF", "standard": "test only", "page": 1,
                 "section": "fixture", "text": "Test fixture, not a production standard."}]
        kb = SimpleNamespace(retrieve=lambda *_args, **_kwargs: refs)
        analyze = Mock(side_effect=lambda *_args, **_kw: {
            "ok": True, "conclusion": "合格", "defect_desc": "test"})
        with tempfile.TemporaryDirectory(prefix="metalqc-web-test-") as temp:
            Path(temp, "perception").mkdir()
            with patch.object(server, "RESULTS_DIR", temp), \
                 patch.object(server, "get_engine", return_value=(engine, config)), \
                 patch.object(server, "get_clf", return_value=classifier), \
                 patch.object(server, "get_sem_kb", return_value=kb), \
                 patch.object(server.vlm_api, "analyze", analyze), \
                 patch.object(report, "save_history") as save_history:
                for score, route in ((.19, "low_risk"), (.2345, "vlm_review"),
                                     (.26, "vlm_review"), (.3, "vlm_review"), (.4, "high_risk")):
                    with self.subTest(score=score):
                        engine.detect.return_value = {"score": score, "heatmap": image,
                                                       "classification": {"正常": .8}}
                        result = self.client.post("/api/detect", files={
                            "file": ("fixture.png", image_bytes.getvalue(), "image/png")},
                            data={"product": "metal_nut"}).json()
                        self.assertTrue(result["ok"], result)
                        self.assertEqual(result["route"], route)
                        self.assertEqual(result["t_high"], .3)
                        self.assertEqual(result["supervised_class"]["label"], "划痕")
                        self.assertEqual(result["perception_prediction"], "defect" if score >= .2345 else "normal")
                        self.assertEqual(result["perception_method"], "CLIP patch 原始基线")
                        if route == "vlm_review":
                            self.assertEqual(result["cognition"]["vlm_conclusion"], "合格")
                            self.assertEqual(result["cognition"]["conclusion"], "需人工复核")
                        from cognition.knowledge_sources import STANDARD_CATALOG
                        self.assertEqual(result["references"], refs + [STANDARD_CATALOG['metal_nut']] if route == "vlm_review" else [])
                        self.assertEqual(save_history.call_args.args[0]["gate"]["t_high"], .3)
                self.assertEqual(analyze.call_count, 3)


if __name__ == "__main__":
    unittest.main()
