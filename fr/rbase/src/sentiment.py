# inside python
import xml.etree.ElementTree as ET
from os.path import join, dirname
from collections import defaultdict

__all__ = ['SentimentRule']

# lexicon and rule based sentiment analysis
class SentimentRule:

    # constructor
    def __init__(self, srcLexicon=None):
        # source lexicon
        self.srcLexicon = srcLexicon
        # lexicon for sentiment
        self.lexicon = defaultdict(defaultdict)

    # load the source lexicon
    def load_lexicon(self):
        # source from pattern
        if self.srcLexicon == 'pattern':
            treeLexicon = ET.parse(join(dirname(__file__), 
                                        'fr-sentiment.xml'))
            root = treeLexicon.getroot()
            for child in root:
                print(child.attrib)
                self.lexicon[child.attrib['form']] = child.attrib
        # no input source
        elif self.srcLexicon == None:
            print('No selected sentiment lexicon')
        # unknown source
        else:
            print('Selected lexicon unknown')

    # score the input text
    def sentiment_score(self, inputText=None):
                


# test
if __name__ == '__main__':
    # class build
    sentimentClf = SentimentRule(srcLexicon='pattern')
    sentimentClf.load_lexicon()

