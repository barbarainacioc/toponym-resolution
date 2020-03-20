*Toponym Resolution with Contextual Word Embeddings*

The requirements file is located in the "system" folder.

To run the toponym resolution system:

python3 run.py --corpora=corpora --embeddingType=embeddingType --geographicInfo=geographicInfo --wiki=wiki 

where the arguments can take the following values:

--corpora : {wotr, lgl}

--embeddingType : {elmo, bert}

--geographicInfo : {yes, no}

--wiki : {yes, no}

To use the ELMo contextual embeddings option it is necessary to pre-install according to its instructions from the repository:
https://github.com/HIT-SCIR/ELMoForManyLangs

In order that the system considers the geographic propreties it is necessary to unzip the files located in the "geoproperties-files" folder.

To use the provided corpora it is necessary to unzip the files located in the "corpora" folder.
