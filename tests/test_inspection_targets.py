from pathlib import Path
import tempfile
import unittest
from products.families import write
from web.family_inspection import inspection_targets
from review.batches import BatchService


class InspectionTargetTests(unittest.TestCase):
    def test_screw_views_collapse_to_one_user_choice_but_keep_internal_recipes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write(root/'product_families/screw/operational_router.json',{'directory':'experiments/router'})
            write(root/'experiments/router/report.json',{'operational_routing_ready':True})
            raw=[{'key':key,'name':key} for key in ('metal_nut','screw','grid','product_a269405b7b9a')]
            targets=inspection_targets(raw,root)
            self.assertEqual([x['key'] for x in targets],['metal_nut','screw','grid']);self.assertEqual(len(raw),4)
            screw=next(x for x in targets if x['key']=='screw');self.assertEqual(screw['name'],'螺丝');self.assertEqual(screw['mode'],'family');self.assertEqual(screw['batch_key'],'family:screw');self.assertTrue(screw['ready'])

    def test_missing_or_escaped_router_never_falls_back_to_fixed_screw_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=[{'key':'screw','name':'fixed'}]
            self.assertFalse(inspection_targets(raw,root)[0]['ready'])
            write(root/'product_families/screw/operational_router.json',{'directory':'../outside'})
            self.assertFalse(inspection_targets(raw,root)[0]['ready'])

    def test_mixed_family_quality_evaluation_is_rejected_before_creating_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            service=BatchService(directory,lambda _:self.fail('Must reject before recipe lookup'),lambda *_:None,lambda _:None)
            try:
                with self.assertRaisesRegex(ValueError,'具体检测配置'):service.create('family:screw','evaluation')
                self.assertEqual(list((Path(directory)/'batches').glob('*/state.json')),[])
            finally:service.executor.shutdown(wait=True)
