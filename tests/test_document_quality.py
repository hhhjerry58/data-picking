import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pytest

from data_picking.openstax_text import cnxml_document, mathml_tex
from data_picking.quality_data import chapter_groups, collection_modules
from data_picking.quality_analysis import match_lengths


def test_math_conversion_keeps_fraction_root_and_subscript_structure():
    root=ET.fromstring('<math><mfrac><msup><mi>x</mi><mn>2</mn></msup><mroot><msub><mi>a</mi><mn>1</mn></msub><mn>3</mn></mroot></mfrac></math>')
    assert mathml_tex(root)==r'\frac{{x}^{2}}{\sqrt[3]{{a}_{1}}}'
    with pytest.raises(ValueError):mathml_tex(ET.fromstring('<math><unknown><mi>x</mi></unknown></math>'))


def test_full_module_keeps_glossary_formulas_and_flags_empty_source_math():
    text,stats=cnxml_document('''<document xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML"><title>Test</title><content><para>Compute <m:math><mi>x</mi></m:math>.</para><media alt="A coordinate graph"><image src="a.png"/></media><m:math><mrow/></m:math></content><glossary><definition><term>Square</term><meaning><m:math><msup><mi>x</mi><mn>2</mn></msup></m:math></meaning></definition></glossary></document>''')
    assert stats['mathml_elements']==stats['formulas']==3
    assert stats['empty_math_elements']==1
    assert '${x}^{2}$' in text and 'Square' in text and 'A coordinate graph' in text


def test_multiscripts_preserve_left_and_right_indices():
    node=ET.fromstring('<math><mmultiscripts><mi>P</mi><mi>r</mi><none/><mprescripts/><mi>n</mi><none/></mmultiscripts></math>')
    assert mathml_tex(node)==r'{}_{n}^{}{P}_{r}^{}'


def test_length_control_uses_common_support_and_never_uses_losses():
    f=pd.DataFrame([{'id':f'{y}:{i}','injected_math_source':y,'source':'math' if y else 'background',
        'group_id':f'{y}:{i}','token_count':100 if i<9 else 10000,'delta_loss':i}
        for y,n in [(1,12),(0,81)] for i in range(n)])
    cfg={'length_matching_seed':42,'matched_background_per_math':9}
    a,summary=match_lengths(f,cfg);f.delta_loss*=-100;b,_=match_lengths(f,cfg)
    assert a.id.tolist()==b.id.tolist()
    assert len(a)==10*a.injected_math_source.sum()
    assert all(r['retained_background']==9*r['retained_math'] for r in summary['bins'])


def test_duplicate_connections_join_openstax_chapter_bootstrap_blocks():
    rows=[{'chapter_id':'a','group_id':'d1'},{'chapter_id':'b','group_id':'d1'},
          {'chapter_id':'b','group_id':'d2'},{'chapter_id':'c','group_id':'d3'}]
    chapter_groups(rows)
    assert rows[0]['group_id']==rows[1]['group_id']==rows[2]['group_id']
    assert rows[3]['group_id']!=rows[0]['group_id']
