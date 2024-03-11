import random
from multiprocessing import cpu_count

from joblib import load
import numpy as np
import pandas as pd
import torch


class Constants:

    def __init__(self):
        pass

    # Пути
    WORKSPACE_PATH = '/content/drive/MyDrive/docs/keepForever/mipt/nlp/hw1_4sem/'
    WORKSPACE_TMP = WORKSPACE_PATH + '/tmp/'
    GIT_HUB_PROJECT_PATH = WORKSPACE_PATH + 'code/'
    WEB_APP_POSTFIX = 'web_app/'
    SRC_POSTFIX = 'src/'
    WEB_APP_SRC_PATH = GIT_HUB_PROJECT_PATH + WEB_APP_POSTFIX + SRC_POSTFIX
    WEB_APP_TEST_PATH = GIT_HUB_PROJECT_PATH + WEB_APP_POSTFIX + 'test/'
    TEST_SCRIPT_PATH = WEB_APP_TEST_PATH + 'test.py'

    ML_POSTFIX = 'ml/'
    ML_SRC_TRAIN_PATH = GIT_HUB_PROJECT_PATH + ML_POSTFIX + SRC_POSTFIX + 'train/'
    DATA_POSTFIX = 'data/'
    RAW_POSTFIX = 'raw/'
    PROCESSED_POSTFIX = 'processed/'
    ASSET_POSTFIX = 'asset/'
    EMBEDDING_POSTFIX = 'embedding/'
    MODEL_POSTFIX = 'model/'
    TOKENIZER_POSTFIX = 'tokenizer/'

    INPUT_DATA_DIR_PATH = GIT_HUB_PROJECT_PATH + ML_POSTFIX + DATA_POSTFIX + RAW_POSTFIX
    PROCESSED_DATA_DIR_PATH = GIT_HUB_PROJECT_PATH + ML_POSTFIX + DATA_POSTFIX + PROCESSED_POSTFIX
    THE_SIMPS_CSV_PATH = INPUT_DATA_DIR_PATH + 'script_lines.csv'

    PROCESSED_QA_PATH = PROCESSED_DATA_DIR_PATH + 'qa_pairs.joblib'

    TARGET_CHAR_PROCESSED_QA_PATH = PROCESSED_DATA_DIR_PATH + 'target_char_qa_pairs.joblib'
    TARGET_CHAR_PROCESSED_ANSWERS_PATH = PROCESSED_DATA_DIR_PATH + 'target_char_answers.joblib'

    TARGET_CHAR_QA_PAIRS_EMBEDDINGS_PATH = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + EMBEDDING_POSTFIX + 'target_char_qa_pairs_embeddings.joblib'
    TARGET_CHAR_ANSWERS_EMBEDDINGS_PATH = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + EMBEDDING_POSTFIX + 'target_char_answers_embeddings.joblib'
    TARGET_CHAR_QA_PAIRS_FAISS_INDEX_PATH = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + EMBEDDING_POSTFIX + 'target_char_qa_pairs_faiss_index.joblib'
    TARGET_CHAR_QA_PAIRS_FAISS_PSA_INDEX_PATH = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + EMBEDDING_POSTFIX + 'target_char_qa_pairs_faiss_psa_index.joblib'

    TARGET_CHAR_QA_PAIRS_EMBEDDINGS_PATH_AUX = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + EMBEDDING_POSTFIX + 'target_char_qa_pairs_embeddings_aux.joblib'
    TARGET_CHAR_ANSWERS_EMBEDDINGS_PATH_AUX = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + EMBEDDING_POSTFIX + 'target_char_answers_embeddings_aux.joblib'
    TARGET_CHAR_QA_PAIRS_FAISS_INDEX_PATH_AUX = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + EMBEDDING_POSTFIX + 'target_char_qa_pairs_faiss_index_aux.joblib'
    TARGET_CHAR_QA_PAIRS_FAISS_PSA_INDEX_PATH_AUX = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + EMBEDDING_POSTFIX + 'target_char_qa_pairs_faiss_psa_index_aux.joblib'

    BI_ENCODER_MODEL_PATH = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + MODEL_POSTFIX + 'bi_encoder_model.pth'
    CROSS_ENCODER_MODEL_PATH = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + MODEL_POSTFIX + 'cross_encoder_model.pth'

    BI_ENCODER_MODEL_PATH_AUX = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + MODEL_POSTFIX + 'bi_encoder_model_aux.pth'
    CROSS_ENCODER_MODEL_PATH_AUX = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + MODEL_POSTFIX + 'cross_encoder_model_aux.pth'

    TOKENIZER_PATH = GIT_HUB_PROJECT_PATH + ML_POSTFIX + ASSET_POSTFIX + TOKENIZER_POSTFIX + 'tokenizer.joblib'

    # Целевой персонаж
    LISA_ID = 9
    LISA_FULL_NAME = 'Lisa Simpson'
    LISA_LC_NAME = 'lisa'

    # Столбцы
    EPISODE_ID_COL = 'episode_id'
    PREV_EPISODE_ID_COL = 'prev_episode_id'
    NUMBER_COL = 'number'

    SPEAKING_LINE_COL = 'speaking_line'
    NORM_TEXT_COL = 'normalized_text'
    RAW_CHAR_TEXT_COL = 'raw_character_text'

    CHAR_ID_COL = 'character_id'
    PREMISE_CHAR_ID_COL = 'premise_char_id'

    LOC_ID_COL = 'location_id'
    PREV_LOC_ID_COL = 'prev_location_id'

    SPOKEN_WORDS_COL = 'spoken_words'
    PREMISE_COL = 'premise'
    PREMISE_UPDATED_COL = 'premise_updated_col'
    TARGET_CHAR_ANSWER_COL = 'target_char_answer_col'

    SAME_LOC_ID_COL = 'same_location_id_dialog'
    SAME_EPISODE_ID_COL = 'same_episode_id'

    RAW_TEXT_COL = 'raw_text'

    LABEL_COL = 'label'
    INVALID_QA_MARK = 0
    VALID_QA_MARK = 1

    # глубина контекста для обучения
    LAG_COUNT = 3

    # столбцы для сортировки
    SIMPS_DF_SORT_BY_COLS = [EPISODE_ID_COL, NUMBER_COL]

    ''' Прочие константы '''

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'DEVICE: {DEVICE}')

    # воспроизводимость
    SEED = 14
    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True  # для воспроизводимости
    torch.backends.cudnn.benchmark = False

    TOKENIZER = load(TOKENIZER_PATH)

    # тренировка
    EPOCHS_OVERFIT_COUNT = 1
    BATCH_SIZE = 16
    MAX_LENGTH = 128

    # вывод
    BI_ENCODER_TOP_N = 8
    GPU_FAISS_INDEX = True
    CROSS_ENCODER_TOP_N = 8
    CROSS_ENCODER_CHUNK_SIZE = 10
    PCA_COMPONENTS_COUNT = 768

    PROC_COUNT = cpu_count()
    print(f"Число процессов для использования: {PROC_COUNT}")

    # логирование
    LOG_FORMAT = 'time="%(asctime)s" level="%(levelname)s" module="%(filename)s" function="%(funcName)s" line=%(lineno)d msg="%(message)s"'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    pd.set_option('display.width', None)  # Автоматическая ширина столбцов
    pd.set_option('display.max_colwidth', None)  # Полное отображение содержимого столбцов
    pd.set_option('display.max_columns', None)  # Полное отображение столбцов (все)
    pd.set_option('display.max_rows', None)  # Полное отобрадение рядов (все)

    # запуск
    GIT_HUB_PROJECT_URL = 'https://github.com/km-mipt-nlp-gen'
    IS_EMBEDDINGS_USED = True
