from llama_index.core.schema import TextNode

import bs4
import logging
import os
from timeit import default_timer as timer

def parse_html_files(file_path):
    text = ""
    with open(file_path, "r") as f:
        soup = bs4.BeautifulSoup(f.read(), features="html.parser")
        # Find all <article> tags
        articles = soup.find_all("article")

        # Extract text from each <article> tag
        for article in articles:
            # Get all text within the <article> tag
            article_text = article.get_text()
            text += article_text + "\n"
    return text


def parse_files_into_nodes(text_parser, dir_path, get_content, extension=".md"):
    nodes = []
    text_parsing_time = 0
    text_splitting_time = 0
    for root, _, files in os.walk(dir_path):
        for filename in files:
            # Check if the file has specified extension
            if filename.endswith(extension):
                # Construct the full path to the file
                start = timer()
                file_path = os.path.join(root, filename)
                page_text = get_content(file_path)
                # logging.info("Parsing file: %s", file_path)
                end = timer()
                text_parsing_time += end - start

                start = timer()
                cur_text_chunks = text_parser.split_text(page_text)

                for _, text_chunk in enumerate(cur_text_chunks):
                    node = TextNode(
                        text=text_chunk,
                    )
                    nodes.append(node)
                end = timer()
                text_splitting_time += end - start

    logging.info(
        f"Text parsing time for {dir_path} {extension} files: {text_parsing_time}s"
    )
    logging.info(
        f"Text splitting time for {dir_path} {extension} files: {text_splitting_time}s"
    )
    return nodes
