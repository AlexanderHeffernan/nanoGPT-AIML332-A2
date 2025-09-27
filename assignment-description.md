# AIML332 Assignment 2: Working with nanoGPT

**Worth:** 15%  
**Due:** 1st October at 5:00pm

In this assignment, you will make modifications to the nanoGPT codebase, and write a report describing your modifications and answering other questions.

For an introduction to nanoGPT, see [my notes here], and [my lecture here]. The notes include a guide for running nanoGPT on Google Colab – but you should be able to do all of this assignment from your own machine, even if it doesn’t have a GPU.

---

## Task 1 (Core, 30 Marks): Exploring Token Probabilities with a Trained GPT Model

Use the pretrained model called `gpt2` (your call to `sample.py` should include `--init_from=gpt2`).

### 1.1. Visualising Token Probabilities

- The trained GPT model generates its response one token at a time. At each iteration, it computes a probability distribution for the next token, then samples this distribution to choose the next token.
- **Write code** that displays a bar chart showing the probability of the top 10 tokens. (MatPlotLib is helpful for this.) If the chart includes the token which is selected, this should be shown in a different colour. Your code should run if the user includes the command-line Boolean flag `show_probs = true`.
- In your report:
  - **(a)** Briefly describe your code, and show some sample output. (Examples using GPT-2 pretrained weights are easiest to interpret.) **(5 marks, including code)**
  - **(b)** How does the value of the command-line argument `temperature` affect token selection? Your answer should show some examples, and also describe how temperature alters the shape of a probability distribution. (NB temperature is a float, not an integer!) **(5 marks)**

### 1.2. Sequence Probability

- Modify the `generate` function in `model.py` so it returns the probability of the complete response it returns: that is, the probability of the generated sequence of tokens.
  - Your code should use log probabilities to do this computation, but return the result as a probability.
- In your report:
  - **(a)** Briefly describe your code. Illustrate the code by reporting the probability for the response to the prompt ‘I live in’, with `max_new_tokens` set to 5, and `temperature` set to 0.0001. **(5 marks, including code)**
  - **(b)** Explain what assumptions you are making in your computation – and why it’s a good idea to perform the computation using log probabilities, to cater for long responses. **(5 marks)**

### 1.3. Probability of a Fixed Sequence

- Further modify the `generate` function so it can take an optional argument specifying the token sequence to be generated. (Your generate function can then be used to return the probability the model assigns to a given input sequence.)
  - The optional argument should be called `fixed_response`. If defined, it should be a list of tokens.
- In your report:
  - **(a)** Briefly describe your code. Illustrate with some example calls. **(5 marks, including code)**
  - **(b)** What is the effect of changing the length of the sequence to be generated? **(5 marks)**

---

## Task 2 (Completion, 40 Marks): Fine-Tuning and Evaluating a GPT Model

Take a pretrained GPT model, and fine-tune it on a dataset of your own choosing. You’ll also create a simple test harness that evaluates the effects of the fine-tuning process.

### 2.1. Evaluation Harness

- Make a new Python program called `eval.py`, that adapts `sample.py` to implement a simple test harness, that evaluates a trained GPT model on a set of prompt-response pairs.
  - `eval.py` should be called with the same arguments as `sample.py`.
  - The prompt-response pairs should be held in a JSON file called `eval_data.json`, structured as a list of prompt-response pairs. (An example file is provided.)
  - A single example pair will be enough for now. (You’ll create a larger set in Question 2.2.)
  - In `eval.py` you should define a function called `eval`, that reads the data file, and calls the `generate` function on each prompt, with the appropriate response as the value of `fixed_response`. The function should print the summed probabilities of each response.
- In your report:
  - **(a)** Briefly describe your code, and show some examples of your evaluation prompt-response pairs. **(10 marks, including code)**
  - **(b)** Briefly describe how your `eval` function runs when `eval.py` is executed. **(5 marks)**

### 2.2. Dataset Selection and Evaluation Set

- Choose a dataset to fine-tune your GPT model on. And define a set of prompt-response pairs, based on this dataset.
  - You can get your text corpus from anywhere – but the datasets on Hugging Face are an obvious place to start. The key requirement is that your dataset contains text of a specific type, or on a specific topic, so that fine-tuning pulls the model towards the kind of text it contains. And your new dataset should be English text.
  - Choose a good way to create prompt-response pairs that are typical for your new dataset, and create a set of prompt-response pairs, to live in `eval_data.json`.
- In your report:
  - **(a)** Describe the corpus you have chosen, and the specialised type of language it contains. **(5 marks)**
  - **(b)** Describe the way you created your evaluation set of prompt-response pairs. You could do this automatically, or by hand, but you should justify your decision either way. (Hand-crafted examples model an ‘alignment’ process, where humans are involved in curating appropriate responses.) **(5 marks, including dataset)**

### 2.3. Fine-Tuning and Evaluation

- ‘Fine-tune’ your GPT-2 model on your selected dataset. To evaluate the success of fine-tuning, run your `eval` function on your evaluation dataset, prior to fine-tuning and after fine tuning.
- In your report:
  - **(a)** Describe the results of your fine-tuning process, with reference to the evaluation dataset. Did it work? If so, how well? **(5 marks)**
  - **(b)** Your evaluation function is a very simple one. What other evaluation methods could you use, to see how well the new dataset has been learned? You should mention two other possible methods, and describe how they work, and what they measure. **(10 marks)**

---

## Task 3 (Challenge, 30 Marks): Describe and Extend nanoGPT

Explore the nanoGPT codebase, by explaining some part of it in your own words, and by extending the system in an interesting direction.

### 3.1. Code Explanation

- Choose some aspect of the code in `model.py` or `train.py` that is distinctive, either in demonstrating how a transformer works, or in demonstrating how PyTorch works.
- Describe the code in a report, around two pages in length. The report should be illustrated with short excerpts from the code, and with traced outputs from the relevant functions. **(15 marks)**

### 3.2. Extension

- Implement an extension of your choice to the nanoGPT code. This is an opportunity to explore something that interests you, or that you’re curious about. Some examples:
  - Implement beam search in the generation function. (The current generator is ‘greedy’, it doesn’t look ahead. We discussed beam search in this lecture.)
  - Implement another evaluation function for the model. (Perhaps based on one of your suggestions for Q2.3(b).)
  - Implement a visualisation tool, to help see what’s happening in some part of the system. For instance, you could create a tool that displays attention weights, or that displays token embeddings (in some lower-dimensional space).
- In your report:
  - **(a)** Describe the new feature you have implemented, and what you think it adds to the system. **(10 marks, including code)**
  - **(b)** Show how the system works, by presenting some output, or examples of use. **(5 marks)**

---

## Submission

- Submit your modified nanoGPT directory as a `.tar` file, and your report as a PDF.
- The submission deadline is Wednesday 1st of October, at 5:00pm.
- You can find information about submission [here]. (This assignment is Assignment_2.)
- You can find information about late submissions on the course Assessments page.

---

## Marking

- Your code will be marked together with the description you provide of it in your report. (These marks are tagged with ‘including code’.)
- To assess code, we may do test runs; we will also consider ease of readability. (The level of comments in Karpathy’s code is a good guide.)
- In your report, we are looking for conciseness and clarity. We’re also looking for evidence that you understand the nanoGPT system, and other concepts you have been taught on the course, if these are relevant.

As with Assignment 1, you are expected to spend around 25 hours on this assignment. Plan your time accordingly!

---

## A Note on AI Use

- You can certainly use AI to help you write code. You can also use AI to help you brainstorm ideas for Task 3.
- Using AI to write components of the report is more problematic, because the report has to reflect your own understanding – including your understanding of the code. (If you didn’t write it, did you really understand it?) We counsel caution here – if you do use AI, make sure it doesn’t impact your understanding.
- To allow us to check on understanding, we may ask you to demonstrate your code and discuss your report in person.
- If you do use AI to help produce any of the content you submit, please include an Appendix in your report, noting how you used it. (This doesn’t apply to spell-checking or style-checking software, or to Web search tools, unless you copy and paste from an ‘AI summary’. But it does include using AI for brainstorming, if you take up any AI-generated suggestions.)