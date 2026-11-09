import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image, ImageFilter

from experiments.audit_screw_coco import load_coco, validate_archive, fingerprint, correspondences
from products.families import write
from review.data_identity import aliases, canonical, reserved_hashes, REGISTRY


class CocoIntakeTests(unittest.TestCase):
    def test_archive_traversal_and_duplicate_members_rejected(self):
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, 'w') as bundle: bundle.writestr('../escape.png', b'bad')
        with zipfile.ZipFile(io.BytesIO(memory.getvalue())) as bundle, self.assertRaises(ValueError):
            validate_archive(bundle)

    def test_unannotated_image_never_becomes_normal(self):
        memory = io.BytesIO(); stream = io.BytesIO()
        Image.new('RGB', (128, 128), 'gray').save(stream, format='PNG')
        document = {'images': [{'id': 0, 'width': 128, 'height': 128, 'file_name': 'image.png'}],
                    'categories': [{'id': 0, 'name': 'Screw-OK'}], 'annotations': []}
        with zipfile.ZipFile(memory, 'w') as bundle:
            bundle.writestr('test/image.png', stream.getvalue())
            bundle.writestr('test/_annotations.coco.json', json.dumps(document))
        with zipfile.ZipFile(io.BytesIO(memory.getvalue())) as bundle:
            samples, _, _ = load_coco(bundle)
        self.assertIsNone(samples[0]['source_label'])
        self.assertFalse(samples[0]['approved_for_training'])

    def test_resize_jpeg_correspondence_and_ambiguous_match(self):
        images = [Image.fromarray(np.random.default_rng(seed).integers(0, 256, (128, 128, 3), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(3)) for seed in (1, 2)]
        refs = [fingerprint(image) for image in images]
        stream = io.BytesIO(); images[0].resize((80, 80)).save(stream, format='JPEG', quality=95)
        with Image.open(io.BytesIO(stream.getvalue())) as image: new = fingerprint(image)
        result = correspondences(np.array([new[1]]), np.array([new[2]]), [{'id': 1}, {'id': 2}],
                                 np.array([r[1] for r in refs]), np.array([r[2] for r in refs]))[0]
        self.assertTrue(result['high_confidence_match'], result)
        self.assertEqual(result['nearest_reference']['id'], 1)
        # Near ties must remain unresolved, even when both distances are tiny.
        ambiguous = correspondences(np.array([new[1]]), np.array([new[2]]), [{'id': 1}, {'id': 2}],
                                    np.array([refs[0][1], refs[0][1]]), np.array([refs[0][2], refs[0][2]]))[0]
        self.assertFalse(ambiguous['high_confidence_match'])
        exact_tie = correspondences(np.array([refs[0][1]]), np.array([refs[0][2]]), [{'id': 1}, {'id': 2}],
                                    np.array([refs[0][1], refs[0][1]]), np.array([refs[0][2], refs[0][2]]))[0]
        self.assertFalse(exact_tie['high_confidence_match'])

    def test_other_product_seals_follow_verified_derivatives_and_cycles_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder)
            write(results/'evaluation_reservations.json', {'products': {'another_product': {'a'*64: {}}}, 'global': {}})
            write(results/REGISTRY, {'schema': 1, 'aliases': {'b'*64: {'canonical_pixel_hash': 'a'*64}, 'c'*64: {'canonical_pixel_hash': 'b'*64}}})
            self.assertEqual(reserved_hashes(results), {'a'*64, 'b'*64, 'c'*64})
            self.assertEqual(canonical('c'*64, aliases(results)), 'a'*64)
            write(results/REGISTRY, {'schema': 1, 'aliases': {'a'*64: {'canonical_pixel_hash': 'b'*64}, 'b'*64: {'canonical_pixel_hash': 'a'*64}}})
            with self.assertRaises(ValueError): reserved_hashes(results)


if __name__ == '__main__': unittest.main()
