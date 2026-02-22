import requests
import logging
import os
import re
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_grobid_raw_xml(pdf_full_path: str, xml_content: str) -> str:
    """
    保存GROBID返回的原始XML内容到文件
    
    Args:
        pdf_full_path: PDF文件的完整路径
        xml_content: GROBID返回的XML内容
    
    Returns:
        str: 保存的XML文件路径
    """
    try:
        # 获取基础文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(pdf_full_path))[0]
        xml_output_path = os.path.join(os.path.dirname(pdf_full_path), f"{base_name}_grobid_raw.xml")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(xml_output_path), exist_ok=True)
        
        # 写入XML文件
        with open(xml_output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        
        logger.info(f"GROBID原始XML已保存至: {xml_output_path}")
        return xml_output_path
    except Exception as e:
        logger.error(f"保存GROBID原始XML时出错: {str(e)}")
        return ""

def test_grobid_connection(endpoint):
    """测试GROBID服务是否正常运行"""
    try:
        # 检查GROBID的存活状态
        health_endpoint = endpoint.replace('/processFulltextDocument', '/isalive')
        response = requests.get(health_endpoint)
        if response.status_code == 200:
            logger.info("GROBID服务正常运行")
            return True
        else:
            logger.error(f"GROBID服务返回状态码: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"无法连接到GROBID服务: {str(e)}")
        return False

def grobid_extract_pdf_to_txt(pdf_directory: str, pdf_filename: str, output_txt_path: str = None) -> bool:
    """
    使用GROBID解析PDF文件并提取文本内容
    
    Args:
        pdf_directory: PDF文件所在的目录路径
        pdf_filename: PDF文件名（如 "paper.pdf"）
        output_txt_path: 输出TXT文件路径（可选，默认在PDF同目录生成同名.txt文件）
    
    Returns:
        bool: 处理是否成功
    """
    GROBID_ENDPOINT = "http://localhost:8071/api/processFulltextDocument"

    # 1) 先检查GROBID服务
    if not test_grobid_connection(GROBID_ENDPOINT):
        return False

    try:
        pdf_full_path = os.path.join(pdf_directory, pdf_filename)
        logger.info(f"处理PDF文件: {pdf_full_path}")

        # 获取基础文件名（不含扩展名）
        base_name, ext = os.path.splitext(pdf_filename)
        if ext.lower() != ".pdf":
            logger.warning(f"传入的文件名不是.pdf格式：{pdf_filename}")

        # 2) 使用GenericLoader和GrobidParser解析PDF
        loader = GenericLoader.from_filesystem(
            pdf_directory,
            glob=pdf_filename,  # 精确匹配该PDF文件
            suffixes=[".pdf"],
            parser=GrobidParser(
                segment_sentences=False,  # 不自动分割句子
                grobid_server=GROBID_ENDPOINT
            )
        )
        
        logger.info("开始加载PDF文档...")
        docs = loader.load()

        # 获取并保存原始XML响应
        if hasattr(loader.parser, 'last_response') and loader.parser.last_response is not None:
            xml_content = loader.parser.last_response.text
            xml_path = save_grobid_raw_xml(pdf_full_path, xml_content)
        else:
            logger.warning("无法获取GROBID原始XML响应")

        if not docs:
            logger.warning("没有从PDF中提取到任何内容")
            return False

        logger.info(f"成功提取了 {len(docs)} 个文档片段")

        # 3) 提取论文标题（优先使用metadata中的paper_title）
        paper_title = next(
            ((d.metadata or {}).get("paper_title") for d in docs
             if (d.metadata or {}).get("paper_title")),
            base_name  # 备用：使用文件名作为标题
        )
        
        # 清理标题，移除特殊字符
        safe_title = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in paper_title).strip()

        # 4) 收集所有文本内容，过滤空内容
        text_chunks = []
        for i, doc in enumerate(docs, 1):
            content = (doc.page_content or "").strip()
            if not content:
                logger.warning(f"文档片段 {i} 内容为空，跳过")
                continue
            text_chunks.append(content)

        if not text_chunks:
            logger.warning("所有文档内容都为空，无法生成文本文件")
            return False

        # 5) 合并所有文本块，添加标题标记
        combined_text = f"[TITLE]{safe_title}[/TITLE]\n" + "\n".join(text_chunks) + "\n"

        # 6) 确定输出文件路径
        if output_txt_path is None:
            output_txt_path = os.path.join(pdf_directory, f"{base_name}.txt")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        
        # 写入文本文件（覆盖模式）
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(combined_text)

        logger.info(f"文本已保存至: {output_txt_path}")
        
        # 预览前100个字符（去除换行）
        preview = combined_text[:100].replace("\n", " ").strip()
        logger.info(f"内容预览: {preview}...")
        
        return True

    except Exception as e:
        logger.error(f"处理PDF时发生错误: {str(e)}")
        return False

def grobid_extract_with_metadata(pdf_path: str) -> dict:
    """
    使用GROBID解析PDF并返回结构化元数据
    
    Args:
        pdf_path: PDF文件的完整路径
    
    Returns:
        dict: 包含标题、作者、摘要等元数据的字典
    """
    GROBID_ENDPOINT = "http://localhost:8071/api/processFulltextDocument"
    
    try:
        pdf_directory = os.path.dirname(pdf_path)
        pdf_filename = os.path.basename(pdf_path)
        
        loader = GenericLoader.from_filesystem(
            pdf_directory,
            glob=pdf_filename,
            suffixes=[".pdf"],
            parser=GrobidParser(
                segment_sentences=False,
                grobid_server=GROBID_ENDPOINT
            )
        )
        
        docs = loader.load()
        
        # 获取并保存原始XML响应
        if hasattr(loader.parser, 'last_response') and loader.parser.last_response is not None:
            xml_content = loader.parser.last_response.text
            save_grobid_raw_xml(pdf_path, xml_content)
        else:
            logger.warning("无法获取GROBID原始XML响应")
        
        if not docs:
            return {}
        
        # 提取元数据
        metadata = {}
        first_doc = docs[0]
        
        if first_doc.metadata:
            metadata = {
                'title': first_doc.metadata.get('paper_title', ''),
                'authors': first_doc.metadata.get('authors', []),
                'abstract': first_doc.metadata.get('abstract', ''),
                'publication_date': first_doc.metadata.get('publication_date', ''),
                'sections': len(docs)
            }
        
        # 获取全文内容
        full_text = "\n".join([doc.page_content for doc in docs if doc.page_content])
        metadata['full_text'] = full_text[:5000] + "..." if len(full_text) > 5000 else full_text  # 限制长度
        
        return metadata
        
    except Exception as e:
        logger.error(f"提取PDF元数据时出错: {str(e)}")
        return {}

# 使用示例
if __name__ == "__main__":
    # 示例用法
    pdf_dir = "/home/kzlab/muse/Savvy/Data_collection/output/3_pdf/2025-09-17/cs.AI"
    pdf_file = "cs.AI_102.pdf"
    
    # 解析PDF并生成TXT文件
    success = grobid_extract_pdf_to_txt(pdf_dir, pdf_file)
    
    if success:
        print("PDF解析成功！")
        
        # 提取元数据
        pdf_path = os.path.join(pdf_dir, pdf_file)
        metadata = grobid_extract_with_metadata(pdf_path)
        print("提取的元数据:", metadata)
    else:
        print("PDF解析失败！")