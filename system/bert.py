import pickle
import string
import re
from pathlib import Path
from ast import literal_eval

import numpy as np
from bert_embedding import BertEmbedding

from utils import get_text_words, normalize_size

def construct_embedding_vector(words, missing, embedding_size, bert):
    embedding = bert([words])
    embedding_vector = np.array(embedding[0][1])

    if missing != 0:
        zero_array = np.zeros((missing, embedding_size), dtype=np.float32)
        embedding_vector = np.concatenate((embedding_vector, zero_array), axis=0)

    return embedding_vector


def bert_embeddings(args, data_file, file_name, batch_size, text_size,
                    mention_size, sentence_size, embedding_size):
    
    bert = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased', max_seq_length=1500, batch_size=32)

    if not Path(file_name).is_file():
        i = 0
        pickle_file = open(file_name, "ab")
        num_lines = sum(1 for line in open(data_file))

        with open(data_file, "r") as file:
            coord_data = []
            mention_data = []
            sentence_data = []
            text_data = []
            translator = str.maketrans(string.punctuation,
                                       ' ' * len(string.punctuation))

            for line in file:

                i += 1
                print("Processing... Instance number: " + str(i))
                
                content = line.split("\t")

                # wiki version has an extra field
                if args.wiki == "yes":
                    coordinates = literal_eval(content[2])
                    mention = content[1].lower().translate(translator)
                    sentence = content[3].lower().translate(translator)
                    text = content[4].lower().translate(translator)

                elif args.wiki == "no":
                    coordinates = literal_eval(content[1])
                    mention = content[0].lower().translate(translator)
                    sentence = content[2].lower().translate(translator)
                    text = content[3].lower().translate(translator)

                coord_data.append(coordinates)

                mention = re.sub("[·–‘’“”«»]", " ", mention)
                mention = mention.rstrip("\n").split(" ")
                mention = list(filter(None, mention))
                    
                sentence = re.sub("[·–‘’“”«»]", " ", sentence)
                sentence = sentence.rstrip("\n").split(" ")
                sentence = list(filter(None, sentence))

                text = re.sub("[·–‘’“”«»]", " ", text)
                text = text.rstrip("\n").split(" ")
                text = list(filter(None, text))

                text_vector, t_miss_sz = get_text_words(mention, sentence, text,
                                                        text_size)
                mention_vector, m_miss_sz = normalize_size(mention,
                                                           mention_size)
                sentence_vector, s_miss_sz = normalize_size(sentence,
                                                            sentence_size)

                mention_final = " ".join(str(x) for x in mention_vector)
                sentence_final = " ".join(str(x) for x in sentence_vector)
                text_final = " ".join(str(x) for x in text_vector)

                mention_embed = construct_embedding_vector(mention_final, m_miss_sz, embedding_size, bert)
                sentence_embed = construct_embedding_vector(sentence_final, s_miss_sz, embedding_size, bert)
                text_embed = construct_embedding_vector(text_final, t_miss_sz, embedding_size, bert)            

                mention_data.append(mention_embed)
                sentence_data.append(sentence_embed)
                text_data.append(text_embed)

                if i % batch_size == 0:
                    pickle.dump([mention_data, sentence_data, text_data],
                                pickle_file)
                    mention_data = []
                    sentence_data = []
                    text_data = []

                if i == num_lines:
                    pickle.dump([mention_data, sentence_data, text_data],
                                pickle_file)

            pickle_file.close()
            file.close()
            return coord_data
    else:
        coord_data = []
        with open(data_file, "r") as file:
            for line in file:
                content = line.split("\t")
                if args.wiki == "yes":
                    coord_data.append(literal_eval(content[2]))
                elif args.wiki == "no":
                    coord_data.append(literal_eval(content[1]))
        return coord_data