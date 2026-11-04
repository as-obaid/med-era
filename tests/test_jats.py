import pytest

from medera.preprocess.jats import article_meta, jats_to_text

BODY = (
    "<sec><title>Introduction</title>"
    '<p>Risk was lower [<xref ref-type="bibr" rid="b1">1</xref>] in women.</p>'
    '<p>Superscript cites<sup><xref ref-type="bibr" rid="b1">1</xref>,<xref ref-type="bibr" rid="b2">2</xref></sup> end.</p>'
    '<p>Real style<xref ref-type="bibr" rid="b1"><sup>1</sup></xref><sup>,</sup>'
    '<xref ref-type="bibr" rid="b2"><sup>2</sup></xref> here.</p>'
    '<p>Range cited (<xref ref-type="bibr" rid="b1">1</xref>–<xref ref-type="bibr" rid="b3">3</xref>) here.</p>'
    '<p>Bracket list [<xref ref-type="bibr" rid="b1">1</xref>, <xref ref-type="bibr" rid="b2">2</xref>].</p>'
    '<table-wrap id="t1"><label>Table 1</label><caption><p>TABLECAPTION</p></caption>'
    "<table><tr><td>TABLECELL</td></tr></table></table-wrap>"
    '<fig id="f1"><label>Figure 1</label><caption><p>FIGCAPTION</p></caption></fig>'
    "<p>Formula <disp-formula><mml:math><mml:mi>FORMULA</mml:mi></mml:math></disp-formula> done.</p>"
    "<p>Inline <inline-formula><mml:math><mml:mi>INLINEMATH</mml:mi></mml:math></inline-formula> math"
    "<inline-formula><tex-math>TEXMATH</tex-math></inline-formula>.</p>"
    "<supplementary-material><p>SUPPLEMENT</p></supplementary-material>"
    "</sec>"
)
BACK = ("<ack><p>ACKNOWLEDGEMENT</p></ack><ref-list><title>REFERENCES</title>"
        '<ref id="b1"><mixed-citation>REFCITATION Smith 2001</mixed-citation></ref></ref-list>')


def test_drops_refs_tables_figs_back_and_citations(make_jats):
    text = jats_to_text(make_jats(body=BODY, back=BACK))
    for gone in ("TABLECAPTION", "TABLECELL", "Table 1", "FIGCAPTION", "Figure 1", "FORMULA", "INLINEMATH",
                 "TEXMATH", "SUPPLEMENT", "ACKNOWLEDGEMENT", "REFERENCES", "REFCITATION",
                 "[1]", "[]", "[, ]", "()", "(–)", "1,2", ".,"):
        assert gone not in text, gone
    for kept in ("Risk was lower in women.", "Superscript cites end.", "Real style here.",
                 "Range cited here.", "Bracket list.", "Formula done.", "Inline math."):
        assert kept in text, kept


def test_drops_back_matter_folded_into_body(make_jats):
    # Europe PMC fullTextXML often has no <back>: back matter arrives as body-level sections.
    body = ("<sec><title>Discussion</title><p>KEEP discussion.</p>"
            "<sec><title>Funding</title><p>KEEP nested funding mention.</p></sec></sec>"
            '<sec id="ack1" sec-type="ack"><title>Acknowledgments</title><p>ACKTEXT</p></sec>'
            '<sec id="notes1"><title>Author Contributions</title><p>AUTHORTEXT</p></sec>'
            "<sec><title>6. Conflicts of Interest</title><p>COITEXT</p></sec>"
            "<sec><title>Funding Statement</title><p>FUNDTEXT</p></sec>"
            "<sec><title>Data Availability Statement</title><p>DATATEXT</p></sec>"
            "<sec><title>Publisher's note</title><p>NOTETEXT</p></sec>"
            '<sec sec-type="fn-group"><title>Footnotes</title><p>FNTEXT</p></sec>'
            '<sec sec-type="contrib-info"><title>Contributor Information</title><p>CONTRIBTEXT</p></sec>'
            '<sec sec-type="ref-list"><title>References</title><ref-list><ref><mixed-citation>REFTEXT'
            "</mixed-citation></ref></ref-list></sec>"
            '<sec sec-type="associated-data"><title>Associated Data</title><p>ASSOCTEXT</p></sec>'
            "<sec><title>Declarations</title><sec><title>Ethics approval and consent to participate</title>"
            "<p>DECLTEXT</p></sec></sec>"
            "<sec><title>CRediT authorship contribution statement</title><p>CREDITTEXT</p></sec>"
            "<sec><title>Supporting information</title><supplementary-material><caption><p>SUPPTEXT</p>"
            "</caption></supplementary-material></sec>"
            "<sec><title>Abbreviations</title><def-list><def-item><term>CI</term><def><p>ABBRTEXT</p></def>"
            "</def-item></def-list></sec>"
            "<sec><title>CONFLICT OF INTEREST STATEMENT</title><p>COI2TEXT</p></sec>"
            "<sec><title>DISCLOSURE OF AI-ASSISTED TECHNOLOGIES IN THE WRITING PROCESS</title><p>AITEXT</p></sec>"
            "<sec><title>Ethical Approval</title><p>ETHICSTEXT</p></sec>"
            "<sec><title>Consent for Publication</title><p>CONSENTTEXT</p></sec>"
            "<sec><title>HUMAN AND ANIMAL RIGHTS</title><p>RIGHTSTEXT</p></sec>")
    text = jats_to_text(make_jats(body=body))
    assert "KEEP discussion." in text and "KEEP nested funding mention." in text and "Funding" in text
    for gone in ("Acknowledgments", "ACKTEXT", "Author Contributions", "AUTHORTEXT", "Conflicts of Interest",
                 "COITEXT", "FUNDTEXT", "DATATEXT", "NOTETEXT", "Footnotes", "FNTEXT", "CONTRIBTEXT",
                 "References", "REFTEXT", "Associated Data", "ASSOCTEXT", "Declarations", "DECLTEXT",
                 "CREDITTEXT", "Supporting information", "SUPPTEXT", "Abbreviations", "ABBRTEXT", "COI2TEXT",
                 "AITEXT", "ETHICSTEXT", "CONSENTTEXT", "RIGHTSTEXT"):
        assert gone not in text, gone


def test_drops_citation_runs_and_author_year_leftovers(make_jats):
    body = ('<p>than men <xref ref-type="bibr" rid="a">[4]</xref>, <xref ref-type="bibr" rid="b">[5]</xref>. '
            "Nevertheless</p>"
            '<p>ever MHT use <xref ref-type="bibr" rid="a">[3]</xref>,<xref ref-type="bibr" rid="b">[10]</xref>,'
            '<xref ref-type="bibr" rid="c">[11]</xref> and four <xref ref-type="bibr" rid="a">[12]</xref>–'
            '<xref ref-type="bibr" rid="b">[14]</xref>.</p>'
            '<p>30 years.<xref ref-type="bibr" rid="a"><sup>3</sup></xref>,<xref ref-type="bibr" rid="b">'
            "<sup>4</sup></xref> In the UK</p>"
            '<p>(Kritsilis et al., <xref ref-type="bibr" rid="a">2018</xref>). Also (Hogervorst et al., '
            '<xref ref-type="bibr" rid="b">2002</xref>; Bagger et al., <xref ref-type="bibr" rid="c">2004</xref>).</p>')
    text = jats_to_text(make_jats(body=body, abstract=""))
    assert text.split("\n\n")[1:] == ["than men. Nevertheless", "ever MHT use and four.", "30 years. In the UK",
                                      "(Kritsilis et al.). Also (Hogervorst et al.; Bagger et al.)."]


def test_blocks_title_abstract_body_in_order(make_jats):
    text = jats_to_text(make_jats(title="My <italic>title</italic>", abstract="<title>Abstract</title><p>Abs.</p>",
                                  body="<sec><title>Methods</title><p>We did it.</p></sec>"))
    assert text == "My title\n\nAbstract\n\nAbs.\n\nMethods\n\nWe did it."


def test_keeps_tail_text_after_removed_xref(make_jats):
    body = '<p>risk was lower<xref ref-type="bibr" rid="b12">12</xref> in women</p>'
    assert jats_to_text(make_jats(body=body, abstract="")).endswith("risk was lower in women")


def test_keeps_non_citation_xrefs_and_parens(make_jats):
    body = ('<p>Shown in <xref ref-type="table" rid="t1">Table 1</xref>; ER(-) and PR(+) tumours'
            " and m<sup>2</sup>.</p>")
    text = jats_to_text(make_jats(body=body))
    assert "Shown in Table 1; ER(-) and PR(+) tumours and m2." in text


def test_keeps_case_numbers_negation(make_jats):
    body = "<p>Women did NOT benefit (HR 0.76, 95% CI)</p>"
    assert "Women did NOT benefit (HR 0.76, 95% CI)" in jats_to_text(make_jats(body=body))


def test_nfkc_and_whitespace(make_jats):
    body = "<p>The ﬁnding   was\n\n \t clear now</p>"
    assert jats_to_text(make_jats(body=body, abstract="")).endswith("The finding was clear now")


def test_nested_paragraphs_not_duplicated(make_jats):
    body = ("<p>Outer start<list><list-item><p>INNER one</p></list-item><list-item><p>INNER two</p>"
            "</list-item></list>outer end.</p>")
    text = jats_to_text(make_jats(body=body))
    assert text.count("INNER one") == 1 and text.count("INNER two") == 1
    assert "Outer start INNER one INNER two outer end." in text


def test_article_meta(make_jats):
    meta = article_meta(make_jats(title="Erratum: <italic>x</italic>", article_type="correction", lang="de"))
    assert meta == {"article_type": "correction", "lang": "de", "title": "Erratum: x"}
    assert article_meta(make_jats(lang=""))["lang"] == ""


def test_unparseable_raises_value_error():
    with pytest.raises(ValueError):
        jats_to_text(b"<article><body>")
    with pytest.raises(ValueError):
        article_meta(b"not xml")
