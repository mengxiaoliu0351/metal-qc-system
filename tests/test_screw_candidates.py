import unittest
from PIL import Image,ImageDraw
from experiments.prepare_screws import aligned_crop,group_splits


class ScrewCandidateTests(unittest.TestCase):
    def test_bbox_row_column_and_rotation_align_one_object(self):
        image=Image.new('RGB',(200,100),'white');draw=ImageDraw.Draw(image);draw.rectangle((80,30,160,50),fill='black')
        crop=aligned_crop(image,[40,120,80,20,0],margin=1.)
        self.assertEqual(crop.size,(20,80));self.assertLess(sum(crop.getpixel((10,40))),30)
        self.assertLess(sum(crop.getpixel((10,5))),30)
        with self.assertRaises(ValueError):aligned_crop(image,[40,120,-1,20,0])

    def test_duplicate_crops_join_source_frames_before_partition(self):
        samples=[{'source_frame':f'{i:03}','pixel_hash':str(i)} for i in range(40)]
        samples[1]['pixel_hash']=samples[0]['pixel_hash']
        samples.append({'source_frame':'000','pixel_hash':'another crop'})
        result=group_splits(samples)
        self.assertEqual(result['groups'],39)
        self.assertEqual(samples[0]['split'],samples[1]['split']);self.assertEqual(samples[0]['split'],samples[-1]['split'])
        self.assertEqual(set(s['split'] for s in samples),{'train','validation','test'})


if __name__=='__main__':unittest.main()
