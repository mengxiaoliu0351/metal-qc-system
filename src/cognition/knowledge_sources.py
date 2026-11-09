"""Project-authored guidance and verified standard catalogue metadata are distinct.

No standard body text or page number is asserted without an original document.
Catalogue identities checked against SAMR on 2026-10-26.
"""
_NOTES = [
    ('图像与适用性', '先核对产品、工艺和采集条件。图像质量或产品适用性异常时重新采集，不依据相对热力颜色认定合格或缺陷。'),
    ('裂纹观察', '对可见的细长开裂线索，记录位置、走向和邻近结构。是否为裂纹及其深度需要进一步核验，单张图像不提供尺寸或报废结论。'),
    ('夹杂与斑点观察', '对局部异色、斑点或颗粒状区域，结合原图核对表面污染、反光或真实缺陷，记录证据后再核验。'),
    ('划痕观察', '对表面线状沟槽或擦伤线索，检查外缘、侧面与关键配合部位，区分加工纹理和损伤。长度、深度和是否允许需实测及真实产品规范。'),
    ('锈蚀与色差观察', '对局部颜色变化，核对光照、表面处理和污染等因素。是否腐蚀及其影响需结合现场证据，不能只按像素颜色下结论。'),
    ('变形观察', '对弯曲、凹陷或形状偏离线索，结合产品图纸、测量和装配要求核验，不从单幅无标尺图像推定公差超限。'),
    ('重新采集', '尽量保持固定视角和采集条件，避开明显过曝、反光与遮挡。图像过暗、近乎空白或不清楚时重新采集。'),
    ('例外核验', '灰区经视觉分析和知识检索后，仅明确且满足作业规则的结论自动分类；仍不确定的灰区、临界分数、证据冲突和周期抽查进入人工任务。最终人工标签独立记录，不覆盖系统原始判断。'),
    ('检验记录', '保留产品、原图、模型版本、系统结果、作业状态及时间。人工核验时另记人员、真实标签和依据，便于追溯误报或漏检。'),
    ('候选迭代', '只有人工确认且批准的标签进入候选训练或校准。候选通过分区对比评估后再决定应用，保留版本回退能力。'),
]
PROJECT_NOTES = [dict(id=f'QC-NOTE-{i:02d}', standard='项目质检知识卡（教学示例，非国标原文）', page=None,
                      section=section, text=text, source_type='project_note', verified_standard=False)
                 for i, (section, text) in enumerate(_NOTES, 1)]

STANDARD_CATALOG = {
    'metal_nut': dict(id='STD-CATALOG-5779-2', standard='GB/T 5779.2-2000', page=None,
        section='紧固件表面缺陷 螺母', source_type='standard_catalog', verified_standard=False,
        source_url='https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=16696E712B9BFBB80C6F8FF46878FC0A',
        text='官方目录名称已核验。本系统尚未接入标准原文；此条目只帮助定位候选标准，不包含条文、页码或合格限值，仍需核对具体产品适用范围。'),
    'screw': dict(id='STD-CATALOG-5779-1', standard='GB/T 5779.1-2000', page=None,
        section='紧固件表面缺陷 螺栓、螺钉和螺柱 一般要求', source_type='standard_catalog', verified_standard=False,
        source_url='https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=67C39C4B7820871074FA0F1660603F3C',
        text='官方目录名称已核验。本系统尚未接入标准原文；此条目只帮助定位候选标准，不包含条文、页码或合格限值，仍需核对具体产品适用范围。'),
}
