import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from products.families import write
from perception.family_router import FamilyRouter


class FamilyRouterTests(unittest.TestCase):
    def test_nearest_unsupported_reference_and_ambiguous_views_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);np.savez_compressed(p/'router.npz',centroids=np.eye(3),exclusions=np.array([[0,-1,0]]))
            meta={'arrays':'router.npz','arrays_sha256':hashlib.sha256((p/'router.npz').read_bytes()).hexdigest(),'quality_decision_permission':False,
                'views':[{'key':s,'name':s} for s in ('screw','head','body')],'gates':[{'similarity':.8,'margin':.1}]*3,'version':'fixture'}
            write(p/'router.json',meta);model=FamilyRouter(p/'router.json')
            self.assertEqual(model.predict([1,0,0])['view_key'],'screw');self.assertFalse(model.predict([0,-1,0])['matched']);self.assertFalse(model.predict([1,1,0])['matched'])
            self.assertFalse(model.predict([1,0,0])['quality_decision_permission'])
            meta['quality_decision_permission']=True;write(p/'router.json',meta)
            with self.assertRaises(ValueError):FamilyRouter(p/'router.json')
