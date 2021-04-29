import spacy

text = "This is Prof. Choi's lecture. His class doesn't start 9:00 A.M. but there are 1,000 examples,pratice,and homeworks."

# Do all preprocessing in a single line
nlp = spacy.load('en_core_web_sm') # Try to add "disable=['parser', 'ner']"
doc = nlp(text)
print(f"{'TOKEN':10} | {'LEMMA':10} | {'POS':7} | {'POS_TAG':7} | {'STOPWORD'}")
print(f"{'-'*10} | {'-'*10} | {'-'*7} | {'-'*7} | {'-'*8}")
for token in doc:
    print(f'{token.text:10} | {token.lemma_:10} | {token.pos_:7} | {token.tag_:7} | {token.is_stop}')

# Visualize spaCy results in a single line
with open('01_preprocess_spacy_dep.svg', 'wt') as f:
    f.write(spacy.displacy.render(doc, style="dep")) # Dependency parse
with open('01_preprocess_spacy_ent.html', 'wt') as f:
    f.write(spacy.displacy.render(doc, style="ent")) # Entity recognition