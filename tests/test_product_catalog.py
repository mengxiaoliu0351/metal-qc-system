import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from products.catalog import ProductCatalog
from products.qualification import catalog_profile
from review.workflow import read, write


class ProductCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.seen_roles = []
        def extract(samples, progress):
            self.seen_roles.extend(s["role"] for s in samples)
            labels = np.array([s["label"] == "defect" for s in samples])
            vectors = np.c_[1 - labels, labels, np.full(len(samples), .1), np.ones(len(samples))].astype(float)
            scores = np.where(labels, .8, .1) + np.arange(len(samples)) * .0001
            return {"bank": np.ones((20, 4)), "global_features": vectors, "clip_scores": scores}
        self.catalog = ProductCatalog(self.root, self.results, extractor=extract)
        self.key = self.catalog.create({"name": "测试螺丝", "sku": "M2", "view": "头部",
            "source": "test fixture", "license": "fixture-only"})["key"]
        self.sequence = 0

    def tearDown(self):
        self.catalog.executor.shutdown(wait=True)
        self.temp.cleanup()

    def sample(self, role="bank", label="normal", **kwargs):
        self.sequence += 1
        image = Image.fromarray(np.random.default_rng(self.sequence).integers(0, 256, (128, 128, 3), dtype=np.uint8))
        stream = io.BytesIO(); image.save(stream, format="PNG")
        self.catalog.add_image(self.key, stream.getvalue(), "sample.png", role, label, **kwargs)
        return stream.getvalue()

    def test_invalid_identity_paths_and_cross_partition_duplicates_are_rejected(self):
        with self.assertRaises(ValueError): self.catalog.detail("../escape")
        image = self.sample()
        self.assertTrue(self.catalog.add_image(self.key, image, "renamed.png", "bank", "normal")["duplicate"])
        with self.assertRaises(ValueError): self.catalog.add_image(self.key, image, "renamed.png", "test", "normal")
        with self.assertRaises(ValueError): self.catalog.add_image(self.key, image, "renamed.png", "bank", "defect")
        self.sample(role="train", group_id="same-part")
        with self.assertRaises(ValueError): self.sample(role="test", group_id="same-part")

    def test_verified_derivatives_cannot_cross_roles_or_reenter_training(self):
        from review.data_identity import REGISTRY
        from review.workflow import image_hash
        source = self.sample('test')
        original = read(self.catalog.folder(self.key)/'product.json')['samples'][0]['pixel_hash']
        with Image.open(io.BytesIO(source)) as image:
            stream = io.BytesIO(); image.save(stream, format='JPEG', quality=80)
        derivative = stream.getvalue()
        with Image.open(io.BytesIO(derivative)) as image:
            import hashlib
            rgb = image.convert('RGB'); fingerprint = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        write(self.results/REGISTRY, {'schema': 1, 'aliases': {fingerprint: {'canonical_pixel_hash': original}}})
        self.assertTrue(self.catalog.add_image(self.key, derivative, 'renamed.jpg', 'test', 'normal')['duplicate'])
        with self.assertRaises(ValueError): self.catalog.add_image(self.key, derivative, 'renamed.jpg', 'bank', 'normal')
        write(self.results/'evaluation_reservations.json', {'global': {}, 'products': {'other_product': {original: {}}}})
        other = self.catalog.create({'name': '另一产品', 'sku': 'fixture', 'view': 'fixture', 'source': 'fixture', 'license': 'fixture'})['key']
        for role in ('bank', 'train', 'calibration', 'qualification', 'feedback'):
            with self.assertRaisesRegex(ValueError, '封存'):
                self.catalog.add_image(other, derivative, 'renamed.jpg', role, 'normal')
        self.catalog.add_image(other, derivative, 'renamed.jpg', 'test', 'normal')

    def test_late_evaluation_reservation_blocks_old_build_activation(self):
        run, state = self.build()
        samples = read(self.catalog.folder(self.key)/'product.json')['samples']
        bank = next(s for s in samples if s['role'] == 'bank')
        write(self.results/'evaluation_reservations.json', {'global': {bank['pixel_hash']: {}}, 'products': {}})
        with self.assertRaisesRegex(ValueError, '封存'): self.catalog.start(self.key)
        with self.assertRaisesRegex(ValueError, '封存'): self.catalog.activate(self.key, run)
        self.assertFalse(self.catalog.registry.exists())

    def build(self, qualified=False):
        for _ in range(10): self.sample()
        for _ in range(10): self.sample("calibration", "normal")
        for _ in range(10): self.sample("calibration", "defect")
        for label in ("normal", "defect"):
            for _ in range(60 if qualified else 2): self.sample("qualification", label)
        self.sample("test", "normal")
        run = self.catalog.start(self.key)["id"]
        self.catalog.futures[run].result(timeout=15)
        return run, self.catalog.detail(self.key)["builds"][0]

    def test_build_excludes_test_images_and_does_not_activate_implicitly(self):
        run, state = self.build()
        self.assertEqual(state["status"], "completed", state)
        self.assertNotIn("test", self.seen_roles)
        self.assertFalse(self.catalog.registry.exists())
        self.assertFalse(state["report"]["automation"]["allow_normal"])
        result = self.catalog.activate(self.key, run)
        profile = catalog_profile(self.results, result["product"])
        self.assertFalse(profile["allow_normal"])
        self.catalog.activate(self.key, run)
        self.assertEqual(len(read(self.catalog.registry)), 1)

    def test_changed_samples_or_model_files_prevent_stale_activation(self):
        run, state = self.build()
        self.assertEqual(state["status"], "completed", state)
        self.sample("test", "defect")
        with self.assertRaisesRegex(ValueError, "样本已变化"):
            self.catalog.activate(self.key, run)

    def test_tampered_bank_and_service_restart_are_detected(self):
        run, state = self.build()
        folder = self.catalog.folder(self.key) / "builds" / run
        config = read(folder / "candidate_config.json")
        (self.results / "perception/banks" / config["bank"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "建库文件已变化"):
            self.catalog.activate(self.key, run)
        state["status"] = "running"; write(folder / "state.json", state)
        self.catalog.futures.clear()
        self.assertEqual(self.catalog.detail(self.key)["builds"][0]["status"], "interrupted")

    def test_enough_independent_qualification_is_required_for_automatic_actions(self):
        run, state = self.build(qualified=True)
        self.assertEqual(state["status"], "completed", state)
        self.assertTrue(state["report"]["automation"]["allow_normal"])
        self.assertTrue(state["report"]["automation"]["allow_isolate"])

    def test_website_api_registers_and_activates_a_product_without_overwriting_registry(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from web import server
        run, state = self.build()
        write(self.catalog.registry, [{"key": "existing", "name": "existing product"}])
        with patch.object(server, "get_product_catalog", return_value=self.catalog), patch.object(server, "reset_product_cache"):
            client = TestClient(server.app)
            listing = client.get("/api/product-catalog").json()
            self.assertEqual(len(listing["products"]), 1)
            response = client.post(f"/api/product-catalog/{self.key}/builds/{run}/activate")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(read(self.catalog.registry)), 2)
            self.assertEqual(read(self.catalog.registry)[0]["key"], "existing")
            invalid = client.post(f"/api/product-catalog/{self.key}/builds", json={"mode": "invalid"})
            self.assertEqual(invalid.status_code, 400)


if __name__ == "__main__": unittest.main()
