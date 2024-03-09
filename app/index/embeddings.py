# Create embeddings for law by OpenAI and Langchain in JSON format

import os
import abc
import time
import logging
import chromadb
from langchain_community.vectorstores import Chroma
from multiprocessing import Pool
from tqdm import tqdm


# Get logger
logger = logging.getLogger(__name__)


# A base class to create embeddings
class Embeddings(metaclass=abc.ABCMeta):
    def __init__(
            self,
            src_filepath: str,
            vectorstore_filepath: str,
            collection_name: str = 'notion',
            chunk_size: int = 800,
            chunk_overlap: int = 10,):
        self.src_filepath = src_filepath
        self.vectorstore_filepath = vectorstore_filepath
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Define percentage of documents to be processed, 100% means all documents.
        # With lower percentage, the processing time will be shorter.
        # This would be useful for debugging or experimenting.
        # Set to 100% for production.
        self.percentage_of_documents_to_be_processed = int(
            os.environ.get('PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED', 1))

    # Abstract function to load documents from source filepath
    @abc.abstractmethod
    def _loader(self) -> list:
        return NotImplementedError

    # Abstract function to split documents into chunked documents
    @abc.abstractmethod
    def _splitter(self, documents: list) -> list:
        return NotImplementedError

    # initial vectorstore with collection names
    def _init_vectorstore(self):
        # create vectorstore
        store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.vectorstore_filepath,
        )
        # persist vectorstore
        store.persist()
        store = None

    # add documents to vectorstore
    def _add_documents(
            self,
            documents: list) -> bool:
        from util.openai import embedder

        return_value = False

        # catch exception to prevent from crashing in multiprocessing
        try:
            # create vectorstore
            store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedder(),
                persist_directory=self.vectorstore_filepath,
            )
            # add documents to vectorstore
            store.add_documents(
                documents=documents,
            )
            # persist vectorstore
            store.persist()
            store = None
            return_value = True
        except Exception as e:
            logger.error(
                f'Add {documents} to {self.collection_name} with error: {e}')

        # debug purpose to list collection names
        vdb = chromadb.PersistentClient(path=self.vectorstore_filepath)
        collection_names = vdb.list_collections()
        logger.debug(f'Collection names: {collection_names}')

        return return_value

    # entry point to run the process
    def run(self):
        from util.openai import calculate_embedding_cost
        from util.tqdm import chunker

        # load documents
        logger.info(f'Loading data from {self.src_filepath}')
        documents = self._loader()
        logger.info(
            f'Loaded {len(documents)} documents from {self.src_filepath}')

        # check length and output the first document for debugging purpose
        if len(documents) > 0:
            logger.debug(f'First document: {documents[0]}')
        else:
            logger.info('No document loaded, exit')
            return

        # cut documents to {self.percentage_of_documents_to_be_processed}%
        if self.percentage_of_documents_to_be_processed < 100:
            logger.info(
                f'Cutting documents to {self.percentage_of_documents_to_be_processed}%')
            documents = documents[:int(
                len(documents)*self.percentage_of_documents_to_be_processed/100)]

        logger.info(f'Ready to process {len(documents)} documents')

        # save documents to file
        logger.info(
            'Saving documents to file notion.documents')
        with open('notion.documents', 'w', encoding='utf-8') as f:
            for document in documents:
                f.write(f'{document}\n')
        # split documents into chunked documents
        logger.info(
            f'Splitting documents into chunked documents with chunk size {self.chunk_size} and overlap {self.chunk_overlap}')
        chunked_documents = self._splitter(documents)
        # output the first chunked document for debugging purpose
        logger.debug(f'First chunked document: {chunked_documents[0]}')

        # save chunked documents to file
        logger.info(
            'Saving chunked documents to file notion.chunked.documents')
        with open('notion.chunked.documents', 'w', encoding='utf-8') as f:
            for document in chunked_documents:
                f.write(f'{document}\n')

        # estimate token and cost
        total_tokens, total_cost = calculate_embedding_cost(documents)
        logger.info(
            f'Total tokens: {total_tokens}, total cost: USD${total_cost:.5f}')

        # ask for confirmation to proceed or not
        proceed = input('Do you want to proceed? (yes/no)')
        if proceed.lower() in ["yes", "y"]:
            logger.info('User confirmed to proceed')
        else:
            logger.info('User cancelled the process')
            return

        # calculate batch size based on total documents
        # batch size is the number of documents to be processed in one batch
        batches, batch_size = chunker(chunked_documents)
        logger.info(
            f'Batch size is {batch_size}, total batches is {len(batches)}')

        # determine number of worker by CPU count, and reserve one for main process
        worker_count = os.cpu_count() - 1
        logger.info(f'Worker count is {worker_count}')

        # create progress bar and prepare to enqueue tasks
        logger.info('Enqueue tasks...(this may take a while)')
        pbar = tqdm(total=len(batches))

        # precreate vectorstore with collection names
        self._init_vectorstore()

        def update_progress(result):
            pbar.update(1)
            # pause 10 mini seconds to avoid too many requests
            time.sleep(0.01)

        def update_error(error):
            logger.error(f'Error: {error}')

        with Pool(processes=worker_count) as pool:
            # enqueue tasks
            for batch in batches:
                pool.apply_async(self._add_documents,
                                 (batch, ),
                                 callback=update_progress,
                                 error_callback=update_error,)
            # close the process pool
            pool.close()
            # wait for all tasks to finish
            pool.join()

            # pause 100 mini seconds to update progress bar
            time.sleep(0.1)

        pbar.close()

        logger.info(f'Batch {len(batches)} processed')


# A class to create embeddings for notion documents
class NotionEmbeddings(Embeddings):
    # custom function to load notion documents
    def _loader(self) -> list:
        from langchain_community.document_loaders import NotionDirectoryLoader

        loader = NotionDirectoryLoader(
            self.src_filepath,
        )

        return loader.load()

    # custom function to split documents into chunked documents
    def _splitter(self, documents: list) -> list:
        from util.openai import markdown_splitter, text_splitter

        md_header_splits = markdown_splitter(
            documents,
            self.chunk_size,
            self.chunk_overlap)

        return text_splitter(
            md_header_splits,
            self.chunk_size,
            self.chunk_overlap,
            separators=["\n\n", "\n", "。", "："])
