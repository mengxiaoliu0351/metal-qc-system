"""Review truth, isolation, candidate publication/rollback and archived PDF checks."""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from review.workflow import ReviewWorkflow, read, write, image_hash, measurement, calibrate
from web import server


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="metalqc-review-")
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.product = "metal_nut"
        self.rng = np.random.default_rng(43)
        self.rows = {}
        def image(path, index):
            path.parent.mkdir(parents=True, exist_ok=True)
            rgb = np.random.default_rng(index).integers(0,256,(12,12,3),dtype=np.uint8)
            Image.fromarray(rgb).save(path)
            return str(path.relative_to(self.root))
        self.image = image
        split = {"seed": 43, "train": [], "validation": [], "test": [], "normal_memory": [], "normal_calibration": []}
        index=0
        for group, count in (("train",24),("validation",16),("test",16)):
            for j in range(count):
                truth=j%2
                path=self.root/f"data/mvtec/metal_nut/test/{'scratch' if truth else 'good'}/{index:03d}.png"
                relative=image(path,index)
                split[group].append(relative)
                feature=np.array([(-2 if not truth else 2)+j*.003,0.1*j/count,0.,0.])
                self.rows[relative]=(feature,truth)
                index+=1
        for j in range(8):
            relative=image(self.root/f"data/mvtec/metal_nut/train/good/{j:03d}.png",100+j)
            split["normal_memory"].append(relative)
        self.split=split
        benchmark=self.results/"experiments/perception_upgrade_v1"
        write(benchmark/"metal_nut_split.json",split)
        paths=split["train"]+split["validation"]+split["test"]
        np.savez_compressed(benchmark/"metal_nut_features.npz",paths=np.array(paths),
            signature=np.array(hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()),
            global_features=np.array([self.rows[p][0] for p in paths]),
            normal_global=np.tile([-2.,0.,0.,0.],(8,1)),
            clip_scores=np.array([.5 if self.rows[p][1] else .1 for p in paths]))
        optimized=self.results/"perception/optimized"
        optimized.mkdir(parents=True)
        np.savez_compressed(optimized/"metal_nut.npz",global_mean=np.zeros(4),global_scale=np.ones(4),global_coef=np.array([[1.,0.,0.,0.]]),global_intercept=np.array([0.]))
        write(optimized/"metal_nut.json",{"product":"metal_nut","mode":"global_binary","name":"fixture","arrays":"metal_nut.npz","threshold":.99,"t_high":.999})
        write(self.results/"perception/banks/products.json",[{"key":"metal_nut","name":"fixture","bank":"bank.npy","threshold":.99,"t_high":.999,"perception_artifact":"metal_nut.json"}])
        np.save(self.results/"perception/banks/bank.npy",np.ones((3,4)))
        self.workflow=ReviewWorkflow(self.root,self.results,extractor=lambda _p,paths:(np.tile([2.,0.,0.,0.],(len(paths),1)),np.full(len(paths),.5)))

    def tearDown(self):
        self.workflow.executor.shutdown(wait=True)
        self.temp.cleanup()

    def record(self, source, session="review", prediction="normal"):
        path=self.results/f"history/2020-01-01/{session}.json"
        target=self.results/f"history/images/{session}.png"
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(self.root/source,target)
        write(path,{"session_id":session,"sample_name":session,"product_key":"metal_nut","product_name":"螺母",
                    "timestamp":"2020-01-01T00:00:00","perception_prediction":prediction,"score":.8,
                    "gate":{"route":"high_risk","threshold":.99,"t_high":.999},
                    "input_png":str(target),"heatmap_png":str(target),
                    "cognition":{"ok":True,"conclusion":"疑似不合格","defect_desc":"测试划痕","suggestion":"人工复核"}})
        return path

    def confirm(self, session="review", verdict="defect", approved=True):
        return self.workflow.confirm({"session_id":session,"verdict":verdict,"reviewer":"测试检验员","use_for_iteration":approved,"note":"划痕<&>核验"})

    def run_candidate(self, mode="model"):
        result=self.workflow.start("metal_nut",mode)
        self.workflow.futures[result["id"]].result(timeout=30)
        return result["id"],read(self.workflow.run_path(result["id"])/"state.json")

    def test_review_compares_original_prediction_preserves_it_and_audits_edits(self):
        path=self.record(self.split["train"][1])
        bank=self.results/"perception/banks/bank.npy"
        before=bank.read_bytes()
        self.confirm()
        result=self.workflow.queue(status="errors")
        self.assertEqual(result["summary"]["false_negative"],1)
        self.assertFalse(result["records"][0]["agreement"])
        self.assertEqual(read(path)["perception_prediction"],"normal")
        self.confirm(verdict="ok",approved=False)
        self.assertEqual(len(list((self.results/"model_iterations/review_audit").glob("*.json"))),2)
        self.assertEqual(self.workflow.approved("metal_nut"),[])
        self.assertEqual(before,bank.read_bytes())

    def test_sealed_evaluation_and_conflicting_dataset_replay_cannot_train(self):
        path = self.record(self.split["train"][1])
        record = read(path)
        record.update(record_scope="evaluation")
        write(path, record)
        with self.assertRaisesRegex(ValueError, "封存评估"): self.confirm()
        self.assertEqual(self.workflow.queue()["count"], 0)
        record.update(record_scope="simulation", source_annotation={"source": "fixture", "label": "defect"})
        write(path, record)
        with self.assertRaisesRegex(ValueError, "来源不一致"): self.confirm(verdict="ok")
        annotation = self.confirm()["annotation"]
        self.assertEqual(annotation["verification_method"], "dataset_label_replay")
        self.assertEqual(annotation["label_source"], "fixture")

    def test_model_candidate_improves_then_publish_and_rollback_are_paired(self):
        self.record(self.split["train"][1]);self.confirm()
        config_before=(self.workflow.config).read_bytes()
        run_id,state=self.run_candidate()
        self.assertEqual(state["status"],"completed",state.get("message"))
        report=state["report"]
        self.assertTrue(report["eligible_for_activation"],report["gate_reasons"])
        self.assertGreater(report["regression"]["candidate"]["accuracy"],report["regression"]["current"]["accuracy"])
        self.assertEqual(self.workflow.config.read_bytes(),config_before)
        self.workflow.activate(run_id)
        live=self.workflow.product("metal_nut")
        meta=read(self.workflow.optimized/live["perception_artifact"])
        self.assertEqual(live["review_version"],run_id)
        self.assertEqual(meta["threshold"],live["threshold"])
        self.workflow.rollback(run_id)
        self.assertEqual(read(self.workflow.config),json.loads(config_before))

    def test_renamed_frozen_test_image_cannot_enter_training(self):
        self.record(self.split["test"][1]);self.confirm()
        _,state=self.run_candidate()
        self.assertEqual(state["status"],"failed")
        self.assertIn("冻结",state["message"])
        self.assertNotIn("review_version",self.workflow.product("metal_nut"))

    def test_real_float32_feature_caches_export_with_runtime_parity(self):
        cache=self.workflow.benchmark/'metal_nut_features.npz'
        with np.load(cache,allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        arrays['global_features']=arrays['global_features'].astype(np.float32)
        arrays['normal_global']=arrays['normal_global'].astype(np.float32)
        np.savez_compressed(cache,**arrays)
        self.record(self.split['train'][1]);self.confirm()
        _,state=self.run_candidate()
        self.assertEqual(state['status'],'completed',state['message'])

    def test_windows_frozen_paths_are_portable_without_changing_signature(self):
        split_path = self.workflow.benchmark / 'metal_nut_split.json'
        split = read(split_path)
        windows = {k: [p.replace('/', '\\') for p in value] if isinstance(value, list) else value for k, value in split.items()}
        write(split_path, windows)
        cache = self.workflow.benchmark / 'metal_nut_features.npz'
        with np.load(cache, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        arrays['paths'] = np.array([p.replace('/', '\\') for p in arrays['paths'].tolist()])
        arrays['signature'] = np.array(hashlib.sha256(json.dumps(windows, sort_keys=True).encode()).hexdigest())
        np.savez_compressed(cache, **arrays)
        self.record(self.split['train'][1]); self.confirm()
        _, state = self.run_candidate()
        self.assertEqual(state['status'], 'completed', state['message'])

    def test_same_pixels_with_conflicting_labels_are_excluded(self):
        self.record(self.split["train"][1],"one");self.confirm("one")
        self.record(self.split["train"][1],"two");self.confirm("two",verdict="ok")
        _,state=self.run_candidate()
        self.assertEqual(state["status"],"failed")

    def test_changed_labels_or_candidate_arrays_prevent_stale_publication(self):
        self.record(self.split["train"][1]);self.confirm()
        run_id,state=self.run_candidate()
        self.assertTrue(state["report"]["eligible_for_activation"])
        self.confirm(verdict="ok")
        with self.assertRaisesRegex(ValueError,"复核标签已变化"):
            self.workflow.activate(run_id)
        # Restore the review signature to isolate artifact-integrity enforcement.
        report=read(self.workflow.run_path(run_id)/"report.json")
        report["review_digest"]=self.workflow.review_digest("metal_nut")
        write(self.workflow.run_path(run_id)/"report.json",report)
        (self.workflow.run_path(run_id)/"candidate.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"发生变化"):
            self.workflow.activate(run_id)

    def test_threshold_mode_requires_independent_feedback_calibration(self):
        self.record(self.split["train"][1]);self.confirm()
        _,state=self.run_candidate("threshold")
        self.assertEqual(state["status"],"failed")
        self.assertIn("校准",state["message"])
        for i in range(300,400):
            source=self.image(self.root/f"external/{i}.png",i)
            fingerprint=image_hash(self.root/source)
            bucket=int(hashlib.sha256(("metal_nut"+fingerprint).encode()).hexdigest()[:8],16)%10
            if bucket in (6,7):break
        self.record(source,"independent");self.confirm("independent")
        _,state=self.run_candidate("threshold")
        self.assertEqual(state["status"],"completed",state["message"])
        self.assertEqual(state["report"]["feedback_usage"]["calibration"],1)

    def test_changed_review_image_prevents_candidate_publication(self):
        record=self.record(self.split['train'][1]);self.confirm()
        run_id,state=self.run_candidate()
        self.assertTrue(state['report']['eligible_for_activation'])
        Image.new('RGB',(12,12),'black').save(read(record)['input_png'])
        with self.assertRaisesRegex(ValueError,'复核图像已变化'):
            self.workflow.activate(run_id)

    def test_new_retained_reservations_block_feedback_training_but_allow_label_storage(self):
        path=self.record(self.split['train'][1]);record=read(path)
        fingerprint=image_hash(record['input_png'])
        write(self.results/'evaluation_reservations.json',{'products':{self.product:{fingerprint:{'experiment':'fixture'}}}})
        with self.assertRaisesRegex(ValueError,'留出评估'):
            self.confirm()
        response=self.workflow.confirm({'session_id':'review','verdict':'defect','reviewer':'fixture','use_for_iteration':False})
        self.assertTrue(response['ok']);self.assertEqual(self.workflow.approved(self.product),[])

    def test_global_sealed_dataset_cannot_reenter_under_another_product(self):
        path=self.record(self.split['train'][1]);fingerprint=image_hash(read(path)['input_png'])
        write(self.results/'evaluation_reservations.json',{'global':{fingerprint:{'dataset':'BODY','role':'test'}}})
        self.assertIn(fingerprint,self.workflow.reserved_images('another-product'))
        with self.assertRaisesRegex(ValueError,'留出评估'):self.confirm()

    def test_api_uses_archived_records_and_exports_human_verdict_pdf(self):
        path=self.record(self.split["train"][1])
        with patch.object(server,"RESULTS_DIR",str(self.results)),patch.object(server,"_review_workflow",self.workflow):
            client=TestClient(server.app)
            r=client.post('/api/review/confirm',json={"session_id":"review","verdict":"defect","reviewer":"测试检验员","use_for_iteration":True,"note":"划痕<&>核验"})
            self.assertEqual(r.status_code,200,r.text)
            self.assertEqual(client.get('/api/review/queue?status=errors').json()["count"],1)
            self.assertEqual(client.get('/api/history/detail',params={"day":"../..","key":"fixture"}).status_code,400)
            response=client.get('/api/history/report',params={"day":path.parent.name,"key":path.stem})
            self.assertEqual(response.status_code,200,response.text[:100] if not response.content.startswith(b'%PDF-') else '')
            self.assertTrue(response.content.startswith(b'%PDF-'))
            import fitz
            with fitz.open(stream=response.content,filetype="pdf") as pdf:
                text=''.join(page.get_text() for page in pdf)
                self.assertIn("人工已确认",text)
            self.assertIn("不一致",text)
            self.assertNotIn("慢链路（深度分析）",text)

    def test_pdf_keeps_local_prediction_separate_from_gray_abstention_and_quality_hold(self):
        from interface.report import generate_pdf
        import fitz
        path=self.record(self.split['train'][1]); record=read(path)
        record.update(perception_prediction='defect',cognition={'conclusion':'需复核','suggestion':'系统已隔离'})
        record['disposition']={'status':'needs_review','label':'待人工复核','requires_review':True,'reason':'当前版本自动处置暂停'}
        record['gate']['route']='vlm_review'
        target=self.results/'report.pdf';generate_pdf(record,str(target))
        with fitz.open(target) as doc:text=''.join(page.get_text() for page in doc)
        self.assertIn('本地整图判定',text);self.assertIn('缺陷',text);self.assertIn('需复核',text)
        record['gate']['route']='high_risk';generate_pdf(record,str(target))
        with fitz.open(target) as doc:text=''.join(page.get_text() for page in doc)
        self.assertIn('当前需人工核验',text);self.assertNotIn('系统已隔离',text)

    def test_calibration_has_both_labels_and_high_risk_constraints(self):
        y=np.array([0]*24+[1]*24);scores=np.r_[np.linspace(.01,.3,24),np.linspace(.4,.95,24)]
        low,high=calibrate(y,scores)
        self.assertLess(low,high)
        m=measurement(y,scores,low,high)
        self.assertEqual(m["accuracy"],1)
        self.assertGreaterEqual(m["high_risk_precision"],.98)
        with self.assertRaises(ValueError):calibrate(np.zeros(5),np.arange(5))


if __name__=='__main__':
    unittest.main()
