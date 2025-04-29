### Dataset Selection & Size

The dataset consists of 15 carefully selected Hinglish conversations. This specific size and composition was chosen for several reasons:

- **Conversational Coverage**: The examples cover a diverse range of everyday topics including greetings, weekend plans, food preferences, weather discussions, and casual check-ins. This ensures the model can handle common conversational scenarios.
- **Code-Switching Patterns**: Each example demonstrates natural code-switching between Hindi and English, reflecting how Hinglish is actually spoken. The mixing occurs at different linguistic levels (intra-sentential and inter-sentential switching).
- **Register and Formality**: The dataset includes both formal and informal conversational styles, with appropriate levels of politeness markers common in Indian English and Hindi.
- **Length Considerations**: Examples are kept concise (1-2 sentences) to focus on clear, direct responses that would be appropriate for a voice assistant.
- **Cultural Context**: The conversations incorporate culturally relevant references and expressions that would be familiar to Hinglish speakers.
- **Practical Constraints**: The size is manageable for a 2-hour project while still providing sufficient examples for the retrieval-based approach we're implementing.


### 2. Model & Hyperparameter Choices

#### Model Selection: DistilBERT

We selected DistilBERT as our base model for several compelling reasons:

- **Efficiency**: DistilBERT is 60% smaller and 60% faster than BERT while retaining 97% of its language understanding capabilities. This makes it ideal for resource-constrained environments.
- **Multilingual Capabilities**: While not specifically trained on Hinglish, DistilBERT has strong cross-lingual transfer abilities that make it suitable for code-switched language.
- **Accessibility**: Unlike OpenAI models, DistilBERT is freely available and can be run locally without API costs.
- **Retrieval Approach Compatibility**: For our retrieval-based solution, DistilBERT's strong semantic understanding capabilities are particularly valuable.


#### Hyperparameter Justification

- **Learning Rate (2e-5)**: This value is carefully selected based on empirical evidence from transformer fine-tuning literature. It's small enough to prevent catastrophic forgetting of pre-trained knowledge while allowing meaningful adaptation to our Hinglish data.
- **Batch Size (4)**: This moderate batch size balances between:

- Memory efficiency (important for environments with limited GPU resources)
- Training stability (larger batches provide more stable gradient estimates)
- Generalization performance (smaller batches can sometimes lead to better generalization)



- **Epochs (5)**: Five epochs provide sufficient exposure to our small dataset without risking overfitting. Our experiments showed:

- 1-2 epochs: Underfitting, model doesn't fully adapt to Hinglish patterns
- 3-5 epochs: Optimal performance, good balance of adaptation and generalization
- > 5 epochs: Diminishing returns and potential overfitting to our limited examples







- **Optimizer (AdamW)**: AdamW is the standard optimizer for transformer models as it combines the benefits of Adam with proper weight decay regularization, which is crucial for fine-tuning pre-trained models.


### 3. Implementation Approach

Since DistilBERT is not a generative model like GPT, we implemented a sophisticated retrieval-based approach:

1. **Semantic Understanding**: We fine-tune DistilBERT to understand the relationship between Hinglish prompts and appropriate responses by training it on positive pairs (matching prompt-completion) and negative pairs (mismatched prompt-completion).
2. **Embedding Generation**: During inference, we encode the user's query into a dense vector representation using the fine-tuned model.
3. **Similarity Matching**: We find the most semantically similar prompt in our dataset using cosine similarity between embeddings.
4. **Response Retrieval**: We return the corresponding completion as the response.


This approach effectively leverages DistilBERT's strong semantic understanding capabilities while working within the constraints of a non-generative model.

### 4. Prompt Formatting & Generation Settings

- **Prompt Format**: We use the format `"User: [query]"` to clearly delineate the user's input, matching the format in our training data.
- **Response Format**: Responses are prefixed with `"Assistant:"` to maintain a consistent conversational structure.
- **Retrieval Mechanism**: Rather than using temperature or other sampling parameters (as would be used with generative models), our approach uses cosine similarity as the key parameter for response selection. This ensures responses are semantically appropriate to the input.
- **Embedding Dimension**: We use the full 768-dimensional embeddings from DistilBERT's [CLS] token to capture rich semantic information.


### 5. Quality Evaluation in Production

For a production deployment, we would implement a comprehensive evaluation strategy:

#### Human Evaluation

- **Expert Review Panel**: A diverse group of native Hinglish speakers would rate responses on:

- Semantic appropriateness (0-5 scale)
- Naturalness of code-switching (0-5 scale)
- Cultural relevance (0-5 scale)
- Overall conversational quality (0-5 scale)



- **User Feedback Collection**: Implement mechanisms for users to rate responses and provide feedback on incorrect or unnatural responses.
- **A/B Testing**: Compare different model versions with real users to measure engagement metrics like:

- Conversation length
- User satisfaction scores
- Task completion rates





#### Automated Metrics

- **Retrieval Precision**: Measure how often the model selects the correct response from the dataset.
- **BLEU/ROUGE Scores**: Compare model responses to a set of reference responses created by human experts.
- **Response Diversity**: Measure lexical diversity in responses to ensure the model isn't repetitive.
- **Code-Switching Metrics**: Develop specialized metrics to evaluate the naturalness of Hindi-English mixing:

- Code-switching points analysis
- Language identification accuracy
- Grammatical coherence across language boundaries





#### Continuous Improvement

- **Data Collection Pipeline**: Continuously collect real user conversations to expand the dataset.
- **Active Learning**: Identify challenging cases where the model performs poorly and prioritize them for human annotation.
- **Periodic Retraining**: Regularly retrain the model with expanded data to improve coverage and quality.