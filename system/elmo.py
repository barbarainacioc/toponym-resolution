from ast import literal_eval
from pathlib import Path
import pickle
import re
import string

import numpy as np
from elmoformanylangs import Embedder

from utils import get_text_words, normalize_size


def elmo_embeddings(args, data_file, file_name, batch_size, text_size,
                    mention_size, sentence_size):
    
    # initailize elmo following ELMoForManyLangs instructions
    elmo = Embedder("144/", batch_size=batch_size)

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
                if i % 10 == 0:
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
                text_vector, m_miss_sz = get_text_words(mention, sentence, text,
                                                        text_size)

                mention_vector, m_miss_sz = normalize_size(mention,
                                                           mention_size)
                sentence_vector, s_miss_sz = normalize_size(sentence,
                                                            sentence_size)

                # Mention
                mention_embed = elmo.sents2elmo([mention_vector])
                mention_embed = np.array(mention_embed[0])
                # Sentence
                sentence_embed = elmo.sents2elmo([sentence_vector])
                sentence_embed = np.array(sentence_embed[0])
                # Text
                text_embed = elmo.sents2elmo([text_vector])
                text_embed = np.array(text_embed[0])

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