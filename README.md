# Hierarchical-Multi-Label-Classification-API

This API can predict text and classify it as hate speech and/or abusive language for **Indonesian Language** because this model only trained by Indonesian Language data. If text classified as hate speech, then it will show hate speech characteristic of that text.
This API is a result of my paper. For reference, you can copy paste **.bib** file that i uploaded or text below.

```bib
@INPROCEEDINGS{Prab1909:Hierarchical,
AUTHOR="Faizal Adhitama Prabowo and Muhammad Okky Ibrohim and Indra Budi",
TITLE="Hierarchical Multi-label Classification to Identify Hate Speech and Abusive
Language on Indonesian Twitter",
BOOKTITLE="2019 6th International Conference on Information Technology, Computer and
Electrical Engineering (ICITACEE) (2019 6th ICITACEE)",
ADDRESS=", Indonesia",
DAYS=25,
MONTH=sep,
YEAR=2019,
KEYWORDS="Hate Seech; Multi-Label Text Classification; Hierarchical Classification;
Machine Learning; RFDT; NB; SVM",
ABSTRACT="Hate speech is one type of speech whose spread is banned in public spaces
such as social media. Twitter is one of the social media used by some
people to broadcast hate speech. The hate speech can be specified based on
the target, category, and level. This paper discusses multi-label text
classification using a hierarchical approach to identify targets, groups,
and levels of speech hate on Indonesian-language Twitter. Identification is
completed using classification algorithms such as the Random Forest
Decision Tree (RFDT), Naïve Bayes (NB), and Support Vector Machine (SVM).
The feature extraction used for classification is the term frequency
feature such as word n-gram and character n-gram. This research conducted
five scenarios with different label hierarchy to find the highest accuracy
that can possibly be reached by hierarchical classification. The
experimental results show that the hierarchical approach with the SVM
algorithm and word uni-gram feature has an accuracy of 68.43\%. It proved
that the hierarchical algorithm can increase data transformation or flat
approach."
}
```

## Example :
Send JSON file to API with text parameter that have value text or array of text as shown in below 
![alt text](https://raw.githubusercontent.com/faizaladhitama/Hierarchical-Multi-Label-Classification-API/master/Capture.JPG "Example")

## Contact
If you have question, please email me at faizaladhitamaprabowo@gmail.com
