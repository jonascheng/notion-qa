import os
import logging
import argparse

# Import proprietory module
import config.env
import config.logging


# Get logger
logger = logging.getLogger(__name__)


# Run the cli app with arguments
if __name__ == '__main__':
    # display type of AI will be used for the app
    logger.info(f'Using AI type: {os.environ.get("OPENAI_API_TYPE")}')

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--create-embeddings',
                        action='store_true',
                        help='create embeddings')
    args = parser.parse_args()

    if args.create_embeddings:
        # Import the module
        from embeddings import NotionEmbeddings

        # Create embeddings
        embeddings = NotionEmbeddings(
            src_filepath=os.environ.get('NOTION_FILEPATH'),
            vectorstore_filepath=os.environ.get('EMBEDDINGS_FILEPATH'),
            collection_name=os.environ.get('EMBEDDINGS_COLLECTION_NAME'),
        )
        embeddings.run()