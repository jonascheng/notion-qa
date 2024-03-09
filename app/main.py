import os
import logging
import argparse

# Import proprietory module
import config.env
import config.logging


# Get logger
logger = logging.getLogger(__name__)


# create embeddings
def create_embeddings():
    # Import the module
    from index.embeddings import NotionEmbeddings

    # Create embeddings
    embeddings = NotionEmbeddings(
        src_filepath=os.environ.get('NOTION_FILEPATH'),
        vectorstore_filepath=os.environ.get('EMBEDDINGS_FILEPATH'),
        collection_name=os.environ.get('EMBEDDINGS_COLLECTION_NAME'),
    )
    embeddings.run()


# Return indexer base on target name
def get_indexer():
    from query.embeddings import QueryEmbeddings

    return QueryEmbeddings(
        vectorstore_filepath=os.environ.get('EMBEDDINGS_FILEPATH'),
        collection_name=os.environ.get('EMBEDDINGS_COLLECTION_NAME'))


# Get relevant documents
def get_relevant_documents_by_query(
        query: str,
        method: str = 'similarity_search'):
    # check if query is empty or string
    if not isinstance(query, str):
        logger.error(f'Query is not a string: {query}')
        return "Please provide a query."

    indexer = get_indexer()

    # similarity search
    if method == 'similarity_search':
        search_results = indexer.similarity_search(query)
    # get relevant documents by retriever
    elif method == 'simple_query':
        retriever = indexer.as_retriever()
        search_results = retriever.invoke(query)
    # elif method == 'multi_query':
    else:
        retriever = indexer.as_multiquery_retriever()
        search_results = retriever.invoke(query)

    return search_results


# Run the cli app with arguments
if __name__ == '__main__':
    # display type of AI will be used for the app
    logger.info(f'Using AI type: {os.environ.get("OPENAI_API_TYPE")}')

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--create-embeddings',
                        action='store_true',
                        help='create embeddings')
    parser.add_argument('--method',
                        type=str,
                        choices=[
                            'similarity_search',
                            'simple_query',
                            'multi_query'],
                        default='similarity_search',
                        help='query method')
    parser.add_argument('--query', type=str, help='query string')
    args = parser.parse_args()

    if args.create_embeddings:
        create_embeddings()
    if args.query:
        search_results = get_relevant_documents_by_query(
            query=args.query,
            method=args.method,)
        print('\n===== Relevant documents =====\n')
        print(search_results)
