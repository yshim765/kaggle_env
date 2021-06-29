from typing import List

import en_core_web_lg
import pysbd
from blingfire import text_to_sentences
from spacy import displacy
from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Whitespace


class DataProcessor():
    def __init__(self) -> None:
        self.nlp = en_core_web_lg.load()
        self.transformers_tokenizer = None
        self.pretokenizer = Whitespace()

    def split_txt2sentence(self, text: str, engine: str = "blingfire") -> List[str]:
        if engine == "blingfire":
            return text_to_sentences(text.replace("\n", "")).split("\n")
        elif engine == "pysbd":
            seg = pysbd.Segmenter(language="en", clean=False)
            return seg.segment(text)
        elif engine == "spacy":
            doc = self.nlp(text)
            return [sentence.text for sentence in doc.sents]
        else:
            raise ValueError("engine value shold be 'blingfire', 'pysbd' or 'spacy'")

    def display_entity_on_jupyter(self, text: str) -> None:
        displacy.render(self.nlp(text), jupyter=True, style='ent')

    def pre_tokenize(self, sentence):
        return self.pretokenizer.pre_tokenize_str(sentence)

    def tokenize(self, text, engine: str = "spacy", engine_settings: dict = {}, reset_transformers_tokenizer: bool = False) -> List[str]:
        if engine == "spacy":
            doc = self.nlp(text)
            return [token.text for token in doc]
        elif engine == "transformers":
            if not bool(engine_settings.get("pretrained_model_name_or_path")):
                raise ValueError("if engine is transformers, pretrained_model_name_or_path is needed in engine_settings.")
            if (not bool(self.transformers_tokenizer)) or reset_transformers_tokenizer:
                self.transformers_tokenizer = AutoTokenizer.from_pretrained(**engine_settings)
            return self.tokenizer(text)
        else:
            raise ValueError("engine value shold be 'spacy' or 'transformers'")

    def find_sublist(self, big_list: List[str], small_list: List[str]) -> List[int]:
        all_positions = []
        for i in range(len(big_list) - len(small_list) + 1):
            if small_list == big_list[i:i + len(small_list)]:
                all_positions.append(i)

        return all_positions

    def tag_sentence_BIO(self, sentences: List[str], label: List[str]) -> List[str]:
        tag = []
        for sentence in sentences:
            nes = ['O'] * len(sentence)
            all_pos = self.find_sublist(sentence, label)
            for pos in all_pos:
                nes[pos] = 'B'
                for i in range(pos + 1, pos + len(label)):
                    nes[i] = 'I'
                
                tag.append(nes)

        return tag
