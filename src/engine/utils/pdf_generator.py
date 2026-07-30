import os
import logging
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa
from typing import Dict, Any

logger = logging.getLogger(__name__)


def link_callback(uri, rel):
    """
    xhtml2pdf에서 로컬 리소스(이미지, 폰트 등)를 찾기 위한 콜백
    """
    # 템플릿 디렉토리 기준
    template_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "templates")
    )

    if uri.startswith("./"):
        path = os.path.join(template_dir, uri[2:])
    else:
        path = os.path.join(template_dir, uri)

    if os.path.isfile(path):
        return path

    # 리소스를 찾지 못했을 때
    return uri


def generate_pdf_report(
    context_data: Dict[str, Any], report_data: Dict[str, str], output_path: str = None
) -> bytes:
    try:
        template_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "templates")
        )
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("report_template.html")

        # HTML 렌더링
        html_out = template.render(context=context_data, report=report_data)

        from io import BytesIO

        result_file = BytesIO()

        pisa_status = pisa.CreatePDF(
            html_out, dest=result_file, encoding="utf-8", link_callback=link_callback
        )

        if pisa_status.err:
            logger.error(f"PDF 생성 에러: {pisa_status.err}")
            return None

        pdf_bytes = result_file.getvalue()

        if output_path:
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)

        return pdf_bytes
    except Exception as e:
        logger.error(f"PDF 생성 중 예외 발생: {e}")
        return None
