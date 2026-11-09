import copy
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from review.batches import BATCH_CONTEXT, BatchService
from review.workflow import read, write


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.results = Path(self.temp.name)
        self.version = {"version": "one"}
        self.contexts = []
        def detect(*args):
            self.contexts.append(dict(BATCH_CONTEXT.get()))
            return {"ok": True, "session_id": "session", "vlm_called": False,
                    "disposition": {"auto_result": "normal", "requires_review": False}}
        self.detect = Mock(side_effect=detect)
        self.service = BatchService(self.results, lambda _: copy.deepcopy(self.version), self.detect, lambda _: None)

    def tearDown(self):
        self.service.executor.shutdown(wait=True)
        self.temp.cleanup()

    def add(self, job, color):
        image = Image.new("RGB", (128, 128), color)
        data = io.BytesIO(); image.save(data, format="PNG")
        return self.service.add(job, data.getvalue(), "fixture.png")

    def test_persistent_batch_deduplicates_and_restores_results(self):
        key = self.service.create("screw")["id"]
        self.add(key, "red")
        self.assertTrue(self.add(key, "red")["duplicate"])
        self.add(key, "blue")
        self.service.start(key); self.service.futures[key].result(5)
        detail = self.service.detail(key)
        self.assertEqual(detail["total"], 2)
        self.assertEqual(detail["counts"]["automatic"], 2)
        self.assertEqual(detail["status"], "completed")
        self.assertTrue(all(c["batch_id"] == key for c in self.contexts))
        self.assertEqual(BATCH_CONTEXT.get(), {})

    def test_failed_only_retry_preserves_completed_items_and_increments_attempt(self):
        self.detect.side_effect = [{"ok": False, "error": "fixture"}, {"ok": True, "session_id": "ok"}]
        key = self.service.create("screw")["id"]
        self.add(key, "red")
        self.service.start(key); self.service.futures[key].result(5)
        self.assertEqual(self.service.detail(key)["counts"]["failed"], 1)
        self.service.start(key, retry_failed=True); self.service.futures[key].result(5)
        detail = self.service.detail(key)
        self.assertEqual(detail["items"][0]["attempt"], 2)
        self.assertEqual(detail["status"], "completed")

    def test_evaluation_callback_only_runs_after_completed_batch(self):
        self.service.evaluate = Mock(return_value={"automatic_release_vetoed": True})
        key = self.service.create("screw", purpose="evaluation")["id"]
        self.add(key, "red")
        self.service.evaluate.assert_not_called()
        self.service.start(key); self.service.futures[key].result(5)
        self.assertEqual(self.service.evaluate.call_args.args[0]["status"], "completed")
        self.assertTrue(self.service.detail(key)["release_evaluation"]["automatic_release_vetoed"])

    def test_restart_recovers_archived_item_without_second_detection(self):
        key = self.service.create("screw", purpose="evaluation")["id"]
        self.add(key, "red")
        item = self.service.items(key)[0]
        item.update(status="running", attempt=1)
        write(self.service.path(key) / "items" / (item["id"] + ".json"), item)
        path = self.service.path(key) / "state.json"
        state = read(path); state["status"] = "running"; write(path, state)
        self.service.executor.shutdown(wait=True)
        recover = Mock(return_value={"ok": True, "session_id": "already_saved"})
        self.service = BatchService(self.results, lambda _: copy.deepcopy(self.version), self.detect, recover)
        self.service.futures[key].result(5)
        self.detect.assert_not_called()
        self.assertEqual(self.service.detail(key)["items"][0]["result"]["session_id"], "already_saved")
        self.assertEqual(recover.call_args.args[0]["record_scope"], "evaluation")

    def test_pause_resume_and_recipe_change_guard(self):
        entered, release = threading.Event(), threading.Event()
        def wait(*_):
            entered.set(); release.wait(5)
            return {"ok": True}
        self.detect.side_effect = wait
        key = self.service.create("screw")["id"]
        self.add(key, "red"); self.add(key, "blue")
        self.service.start(key); self.assertTrue(entered.wait(5))
        self.service.pause(key); release.set(); self.service.futures[key].result(5)
        self.assertEqual(self.service.detail(key)["counts"]["completed"], 1)
        self.assertEqual(self.service.detail(key)["status"], "paused")
        self.version["version"] = "two"
        with self.assertRaisesRegex(ValueError, "配置已变化"): self.service.start(key)
        self.service.cancel(key)
        self.assertFalse(self.service.active())

    def test_temporarily_unavailable_recipe_pauses_without_losing_completed_items(self):
        recipe=self.service.recipe
        def detect(*args):
            if self.detect.call_count==1:self.service.recipe=Mock(side_effect=ValueError('router unavailable'))
            return {'ok':True,'session_id':'checked'}
        self.detect.side_effect=detect;key=self.service.create('family:screw')['id']
        self.add(key,'red');self.add(key,'blue');self.service.start(key);self.service.futures[key].result(5)
        result=self.service.detail(key);self.assertEqual(result['status'],'paused');self.assertEqual(result['counts']['completed'],1)
        self.service.recipe=recipe;self.service.start(key);self.service.futures[key].result(5)
        self.assertEqual(self.service.detail(key)['counts']['completed'],2);self.assertEqual(self.detect.call_count,2)

    def test_invalid_ids_and_empty_batches_do_not_schedule_work(self):
        with self.assertRaises(ValueError): self.service.detail("../../escape")
        key = self.service.create("screw")["id"]
        with self.assertRaises(ValueError): self.service.start(key)
        self.assertEqual(self.detect.call_count, 0)

    def test_catalog_test_import_is_idempotent_and_labels_are_not_in_model_context(self):
        from review.workflow import image_hash
        source = self.results / "source.png"
        Image.new("RGB", (128, 128), "green").save(source)
        rows = [{"path": "source.png", "pixel_hash": image_hash(source), "label": "defect", "original_name": "fixture.png"}]
        model_contexts = []
        def inspect(*args):
            model_contexts.append(dict(BATCH_CONTEXT.get()))
            return {"ok": True, "perception_prediction": "normal",
                    "disposition": {"auto_result": "normal", "requires_review": False}}
        self.detect.side_effect = inspect
        job = self.service.from_catalog("screw", rows, self.results, "public fixture", "evaluation")
        self.service.futures[job["id"]].result(5)
        repeated = self.service.from_catalog("screw", rows, self.results, "public fixture", "evaluation")
        self.assertEqual(job["id"], repeated["id"])
        result = self.service.detail(job["id"])
        self.assertEqual(result["evaluation"]["local_false_negatives"], 1)
        self.assertEqual(result["evaluation"]["automatic_normal_misses"], 1)
        self.assertNotIn("import_rows", result)
        self.assertEqual(len(model_contexts), 1)
        self.assertNotIn("dataset_truth", model_contexts[0])
        self.assertNotIn("source_annotation", model_contexts[0])

    def test_evaluation_records_do_not_enter_operational_quality_counts(self):
        from review.metrics import operational_summary
        record = {"record_scope": "evaluation", "time_total": 1,
                  "disposition": {"auto_result": "normal", "requires_review": False}}
        summary = operational_summary([record])
        self.assertEqual(summary["new_policy_records"], 0)
        self.assertEqual(summary["automatic_decisions"], 0)

    def test_website_batch_api_runs_and_serves_the_persisted_source_image(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from web import server
        with patch.object(server, "get_batch_service", return_value=self.service):
            client = TestClient(server.app)
            response = client.post("/api/batches", json={"product": "screw"})
            self.assertEqual(response.status_code, 200, response.text)
            key = response.json()["id"]
            image = io.BytesIO(); Image.new("RGB", (128, 128), "blue").save(image, format="PNG")
            upload = client.post(f"/api/batches/{key}/items", files={"file": ("fixture.png", image.getvalue(), "image/png")})
            self.assertEqual(upload.status_code, 200, upload.text)
            start = client.post(f"/api/batches/{key}/start", json={})
            self.assertEqual(start.status_code, 200, start.text)
            self.service.futures[key].result(5)
            detail = client.get(f"/api/batches/{key}").json()
            self.assertEqual(detail["counts"]["completed"], 1)
            item = detail["items"][0]["id"]
            photo = client.get(f"/api/batches/{key}/items/{item}/image")
            self.assertEqual(photo.status_code, 200)
            self.assertEqual(photo.headers["content-type"], "image/png")


if __name__ == "__main__": unittest.main()
