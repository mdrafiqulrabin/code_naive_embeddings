# code_naive_embeddings

This repository contains the code for obtaining naive embeddings (i.e., sequence of characters/tokens) and training the sequence-based Multilayered Bidirectional GRU model.

---

## Structure

```
├── config.py          # Define path/parameters for data/model
├── helper.py          # Helper fucntions: log/track/vocab
├── main.py            # Load data, Create Vocabulary, and Run Model 
├── model_handler.py   # Multilayered Bidirectional GRU model
├── token_encoder.py   # Input encoder: OneHot/GloVe/MethodToken
``` 

---

### Related Work: https://github.com/mdrafiqulrabin/handcrafted-embeddings

