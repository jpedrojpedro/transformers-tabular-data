from torch import nn
from transformers import AutoTokenizer,              \
                         T5Tokenizer,                \
                         AutoModelForSeq2SeqLM,      \
                         T5ForConditionalGeneration, \
                         AutoModelWithLMHead
# from torchsummary import summary


class TabularT0(nn.Module):
    MODEL_NAME = 'bigscience/T0_3B'

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = AutoModelForSeq2SeqLM.from_pretrained(TabularT0.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(TabularT0.MODEL_NAME)

    def forward(self, **kwargs):
        preds = self.model(**kwargs)
        return preds.logits

    def generate(self, X):
        return self.model.generate(X)


class TabularT5(nn.Module):
    MODEL_NAME = 't5-small'

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = T5ForConditionalGeneration.from_pretrained(TabularT5.MODEL_NAME)
        self.tokenizer = T5Tokenizer.from_pretrained(TabularT5.MODEL_NAME)

    def forward(self, **kwargs):
        preds = self.model(**kwargs)
        return preds.logits

    def generate(self, X):
        return self.model.generate(input_ids=X)


class TabularT5QA(nn.Module):
    MODEL_NAME = 'mrm8488/t5-base-finetuned-qasc'

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = AutoModelWithLMHead.from_pretrained(TabularT5QA.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(TabularT5QA.MODEL_NAME)
        self.context = None

    def forward(self, **kwargs):
        preds = self.model(**kwargs)
        return preds.logits

    def get_response(self, question, max_length=64):
        input_text = "question: {}  context: {}".format(question, self.context)
        features = self.tokenizer([input_text], return_tensors='pt')
        output = self.model.generate(
            input_ids=features['input_ids'],
            attention_mask=features['attention_mask'],
            max_length=max_length
        )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)

    def build_context(self, *facts):
        self.context = ' '.join(facts)


if __name__ == '__main__':
    model = TabularT5QA(3)
    statements = [
        'a watch is used for measuring time',
        'times are measured in seconds',
    ]
    model.build_context(*statements)
    question = 'What can be used to measure seconds? ' \
               '(A) Watch (B) seconds (C) fluid (D) Ruler (E) goggles (F) glasses (G) Drill (H) Scale'
    print(model.get_response(question))
    # task = 'Reorder the words in this sentence'
    # sentences = [
    #     'justin and name bieber years is my am I 27 old.',
    #     'table the is the book on'
    # ]
    # x = model.tokenizer.encode(
    #     ["{}: {}".format(task, sentence) for sentence in sentences], return_tensors='pt', padding=True
    # )
    # y = model.generate(x)
    # print(model)
    # summary(model, x)
    # print(model.tokenizer.batch_decode(y, skip_special_tokens=True))
