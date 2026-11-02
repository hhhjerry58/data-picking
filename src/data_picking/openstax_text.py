"""Conservative CNXML to readable text, preserving structured MathML formulas."""
from collections import Counter
import re
import xml.etree.ElementTree as ET

MATH = "{http://www.w3.org/1998/Math/MathML}"
CNX = "{http://cnx.rice.edu/cnxml}"


def tex_text(text):
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_",
                    "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "^": r"\^{}", "~": r"\~{}"}
    return "".join(replacements.get(c, c) for c in text)


def mathml_tex(node):
    tag = node.tag.split("}")[-1]
    children = list(node)
    values = [mathml_tex(c) for c in children]
    def arity(n):
        if len(values) != n: raise ValueError(f"Invalid MathML arity: {tag} {len(values)} != {n}")
    if tag in {"math", "mrow", "mstyle", "mpadded", "mtd"}:
        result = " ".join(values)
        if node.attrib.get("mathvariant") == "bold" or node.attrib.get("fontweight") == "bold":
            result = r"\mathbf{"+result+"}"
        return result
    if tag in {"mi", "mn", "mo", "mtext", "ms"}:
        if children: raise ValueError("Nested mathematical token elements")
        text = re.sub(r"\s+", " ", node.text or "").strip()
        if not text and tag != "mtext": return ""
        if tag in {"mtext", "ms"}: return r"\text{"+tex_text(text)+"}"
        if tag == "mo":
            return {"{": r"\{", "}": r"\}", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$",
                    "_": r"\_", "\\": r"\backslash", "⁡": r"\,", "⁢": r"\,", "⁣": ","}.get(text, text)
        if node.attrib.get("mathvariant") == "normal": return r"\mathrm{"+tex_text(text)+"}"
        if node.attrib.get("mathvariant") == "bold": return r"\mathbf{"+tex_text(text)+"}"
        return tex_text(text)
    if tag == "mfrac":
        arity(2)
        if node.attrib.get("linethickness") in {"0", "0px", "0pt"}:
            return r"\genfrac{}{}{0pt}{}{"+values[0]+"}{"+values[1]+"}"
        return r"\frac{"+values[0]+"}{"+values[1]+"}"
    if tag in {"msup", "msub"}:
        arity(2); return "{"+values[0]+"}"+("^" if tag == "msup" else "_")+"{"+values[1]+"}"
    if tag == "msubsup":
        arity(3); return "{"+values[0]+"}_{"+values[1]+"}^{"+values[2]+"}"
    if tag == "msqrt": return r"\sqrt{"+" ".join(values)+"}"
    if tag == "mroot":
        arity(2); return r"\sqrt["+values[1]+"]{"+values[0]+"}"
    if tag == "mfenced":
        left, right = node.attrib.get("open", "("), node.attrib.get("close", ")")
        sep = node.attrib.get("separators", ",").replace(" ", "")
        body = "".join(v+(sep[min(i,len(sep)-1)] if sep and i<len(values)-1 else "") for i,v in enumerate(values))
        return {"{": r"\{"}.get(left,left)+body+{"}": r"\}"}.get(right,right)
    if tag in {"mover", "munder"}:
        arity(2); command = r"\overset" if tag == "mover" else r"\underset"
        return command+"{"+values[1]+"}{"+values[0]+"}"
    if tag == "munderover":
        arity(3); return r"\overset{"+values[2]+r"}{\underset{"+values[1]+"}{"+values[0]+"}}"
    if tag == "mtable": return r"\begin{matrix}"+r" \\ ".join(values)+r"\end{matrix}"
    if tag in {"mtr", "mlabeledtr"}: return " & ".join(values)
    if tag == "mspace": return r"\ "
    if tag == "menclose":
        return r"\enclose{"+node.attrib.get("notation", "longdiv")+"}{"+" ".join(values)+"}"
    if tag == "mphantom": return r"\phantom{"+" ".join(values)+"}"
    if tag in {"none", "mprescripts"}: return ""
    if tag == "mmultiscripts":
        if not values: raise ValueError("Empty multiscript")
        left, right, current = [], [], []
        for child, value in zip(children[1:], values[1:]):
            if child.tag.split("}")[-1] == "mprescripts": right = current; current = []
            else: current.append(value)
        if any(c.tag.split("}")[-1] == "mprescripts" for c in children): left = current
        else: right = current
        if len(left)%2 or len(right)%2: raise ValueError("Unpaired multiscripts")
        def scripts(v): return "".join("_{"+v[i]+"}^{"+v[i+1]+"}" for i in range(0,len(v),2))
        return ("{}"+scripts(left) if left else "")+"{"+values[0]+"}"+scripts(right)
    raise ValueError(f"Unsupported MathML element: {tag}")


def cnxml_document(xml):
    root = ET.fromstring(xml)
    title = root.findtext(CNX+"title") or "Untitled"
    content = root.find(CNX+"content")
    if content is None: raise ValueError("CNXML document has no content")
    stats = Counter()
    bodies = [content]+root.findall(CNX+"glossary")
    expected = sum(e.tag == MATH+"math" for b in bodies for e in b.iter())
    ids = {e.attrib["id"]: e.tag.split("}")[-1] for b in bodies for e in b.iter() if "id" in e.attrib}
    def normalized(text): return re.sub(r"\s+", " ", text or "")
    def children(node):
        return normalized(node.text)+"".join(render(c)+normalized(c.tail) for c in node)
    def render(node):
        if node.tag == MATH+"math":
            stats["formulas"] += 1
            value = mathml_tex(node)
            if not value.strip(): stats["empty_math_elements"] += 1
            return ("\n$$ "+value+" $$\n") if node.attrib.get("display") == "block" else "$"+value+"$"
        if node.tag.startswith(MATH): raise ValueError("MathML element outside math wrapper")
        tag = node.tag.split("}")[-1]
        if tag == "media":
            stats["images"] += 1
            alt = normalized(node.attrib.get("alt", "")).strip()
            if not alt: stats["images_without_description"] += 1
            return "\n[Figure: "+(alt or "image without text description")+"]\n"
        if tag in {"image", "colspec"}: return ""
        body = children(node)
        if tag == "title": return "\n## "+body.strip()+"\n"
        if tag == "link" and not body.strip():
            target = node.attrib.get("target-id")
            return "["+ids.get(target, "reference")+"]" if target else ""
        if tag == "entry": return body.strip()+" | "
        if tag == "row": return "\n"+body.strip()+"\n"
        if tag == "item": return "\n- "+body.strip()+"\n"
        if tag in {"solution", "problem", "example"}: return "\n"+tag.capitalize()+":\n"+body.strip()+"\n"
        if tag in {"sup", "sub"}: return ("^(" if tag=="sup" else "_(")+body+")"
        if tag in {"newline", "space"}: return "\n" if tag=="newline" else " "
        if tag in {"section", "para", "exercise", "note", "figure", "table", "caption", "definition", "list", "glossary"}:
            return "\n"+body.strip()+"\n"
        return body
    text = "# "+title+"\n\n"+"\n\n".join(render(b) for b in bodies)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if stats["formulas"] != expected: raise ValueError("Some MathML formulas were not visited")
    return text, {"title": title, "mathml_elements": expected, **dict(stats)}
