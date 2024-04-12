from llama_index.core.schema import TextNode

import bs4
import logging
import os


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
    for root, _, files in os.walk(dir_path):
        for filename in files:
            # Check if the file has specified extension
            if filename.endswith(extension):
                # Construct the full path to the file
                file_path = os.path.join(root, filename)
                page_text = get_content(file_path)
                logging.info("Parsing file: %s", file_path)

                cur_text_chunks = text_parser.split_text(page_text)

                for _, text_chunk in enumerate(cur_text_chunks):
                    node = TextNode(
                        text=text_chunk,
                    )
                    nodes.append(node)

    return nodes
