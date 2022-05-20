# notes for rwkv2 integration

* Detailed HF guide: https://huggingface.co/docs/transformers/add_new_model
* GPT2 example by Thomas: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28

Steps:

 1. [ ] (Optional) Understood theoretical aspects
 2. [ ] Prepared transformers dev environment
 3. [ ] Set up debugging environment of the original repository
 4. [ ] Created script that successfully runs forward pass using original repository and checkpoint
 5. [ ] Successfully added the model skeleton to Transformers
 6. [ ] Successfully converted original checkpoint to Transformers checkpoint
 7. [ ] Successfully ran forward pass in Transformers that gives identical output to original checkpoint
 8. [ ] Finished model tests in Transformers
 9. [ ] Successfully added Tokenizer in Transformers
10. [ ] Run end-to-end integration tests
11. [ ] Finished docs
12. [ ] Uploaded model weights to the hub
13. [ ] Submitted the pull request
14. [ ] (Optional) Added a demo notebook

Style notes:

* The forward pass of your model should be fully written in the modeling file while being fully independent of other models in the library. If you want to reuse a block from another model, copy the code and paste it with a # Copied from comment on top (see here for a good example).
* The code should be fully understandable, even by a non-native English speaker. This means you should pick descriptive variable names and avoid abbreviations. As an example, activation is preferred to act. One-letter variable names are strongly discouraged unless it’s an index in a for loop.
* More generally we prefer longer explicit code to short magical one.
* Avoid subclassing nn.Sequential in PyTorch but subclass nn.Module and write the forward pass, so that anyone using your code can quickly debug it by adding print statements or breaking points.
* Your function signature should be type-annotated. For the rest, good variable names are way more readable and understandable than type annotations.

