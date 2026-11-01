import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cognition.rag import KnowledgeBase, SemanticKnowledgeBase
from cognition.knowledge_sources import PROJECT_NOTES, STANDARD_CATALOG
from cognition.vlm_api import build_prompt, SYSTEM_PROMPT

class KnowledgeProvenanceTests(unittest.TestCase):
    def test_unverified_notes_never_masquerade_as_standard_originals(self):
        self.assertEqual(KnowledgeBase().retrieve('划痕')[0]['id'], 'QC-NOTE-04')
        self.assertTrue(all(f['page'] is None and f['source_type'] == 'project_note' and not f['verified_standard'] for f in PROJECT_NOTES))
        self.assertTrue(all(not f['id'].startswith('GB34528') for f in PROJECT_NOTES))
        for kb in (KnowledgeBase(), SemanticKnowledgeBase(lambda texts: np.ones((len(texts), 4)))):
            refs = kb.retrieve('划痕', top_k=3)
            self.assertTrue(all(r['source_type'] == 'project_note' and r['page'] is None for r in refs))
        self.assertIn('不是国标原文', build_prompt('划痕', ['[QC-NOTE-04] fixture']))
        self.assertIn('不能量化', SYSTEM_PROMPT)
        self.assertEqual(STANDARD_CATALOG['metal_nut']['standard'], 'GB/T 5779.2-2000')
        self.assertTrue(STANDARD_CATALOG['metal_nut']['source_url'].startswith('https://openstd.samr.gov.cn/'))
