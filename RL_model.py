


import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.util import ngrams
nltk.download('punkt')
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy.symbols import nsubj, VERB, NOUN


class Environment:

    def __init__(self, p1, text, words, pars):
        self.board=[]#[word_tag, index-1_tag, index+1_tag,stopword, tf, Reward]
        self.p1 = p1
        self.text = text
        self.Ngramwords = words
        self.Ncount= len(words)
        self.pars = pars
        self.isEnd = False
        self.boardHash = None
        self.wordlength=np.zeros(len(self.Ngramwords), dtype = int)
        self.positiveSymbol = 1
        self.negativeSymbol = -1
        self.PARtag=[]
        self.Btag=[]
        self.Atag=[]
        self.rwd=[]


   

    def availablePositions(self):
       positions = []
       for i in range(len(self.Ngramwords)):
           if self.wordlength[i] == 0:
               positions.append(i)
       return positions

    def tags (self, par):
        doc = nlp(self.text.lower())
        for token in doc:
          if token.text == par.lower():
            if token.pos == NOUN:
              return 1
            elif token.pos == VERB:
              return 2
            else:
              return 3
        
    def updateState(self, position):
        Btag=position-1
        Atag=position+1 
        tagnumber=self.tags(self.Ngramwords[position])
        self.PARtag.append(tagnumber)
        if Atag <= (self.Ncount-1):
          Atagnumber=self.tags(self.Ngramwords[Atag])
          self.Atag.append(Atagnumber)
        else:
          self.Atag.append(4)
                 
        if Btag >= 1:
          Btagnumber=self.tags(self.Ngramwords[Btag])
          self.Btag.append(Btagnumber) 
        else:
          self.Btag.append(5)
                  
        
        if self.Ngramwords[position] in self.pars:
            self.wordlength[position] = self.positiveSymbol
            self.rwd.append(self.positiveSymbol)
           
        else:
            self.wordlength[position] = self.negativeSymbol
            self.rwd.append(self.negativeSymbol)
           

    def finaltag(self):
        finalPARtag.extend(self.PARtag)
        finalBtag.extend(self.Btag)
        finalAtag.extend(self.Atag)
        finalrwd.extend(self.rwd)

    def CurationDone(self):
        if len(self.availablePositions()) == 0:
          self.isEnd = True
          return 0
        # not end
        self.isEnd = False
        return None

  

    def play(self):# rounds = number of sentences in dataset
        while not self.isEnd:
          positions = self.availablePositions()
          p1_action = self.p1.chooseAction(positions)
          # take action and upate board state
          self.updateState(p1_action)
          self.board=[]
          # check board status if it is end
          Done = self.CurationDone()
          if Done is not None:
            break  

      

              
            


class Agent:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        #print("self.name",self.name)
        self.states = pd.DataFrame(columns = ["tag", "Btag","Atag","Reward"])  # record all positions taken
        self.exp_rate = exp_rate


    def extract_ngrams(self, data, num):
        n_grams = ngrams(nltk.word_tokenize(data), num)
        Ngramwords=[ ' '.join(grams) for grams in n_grams]
        return Ngramwords

    def chooseAction(self, positions):

       # if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
        idx = np.random.choice(len(positions))
        action = positions[idx]
        
        return action

    # append a hash state
    def addState(self, state):
        #print("state", state)
        self.states=state
        #print(self.states)
        
        
    def updateStatetags(self, tag, btag, atag, rewd):
        df2=pd.DataFrame()
        df2["tag"]=tag
        df2["Btag"]=btag
        df2["Atag"]=atag
        df2["Reward"]=rewd
        #print(df2.tail(5))
        #print(df2.shape)
        return df2

    def reset(self):
        self.states = []


    def savePolicy(self, df):
        fw = open('test1_' + str(self.name), 'wb')
        pickle.dump(df, fw)
        #print(self.states)
        fw.close()
        #print(self.states)

    
if __name__ == "__main__":
     # Function to generate n-grams from sentences.
    finalPARtag=[]
    finalBtag=[]
    finalAtag=[]
    finalrwd=[]    
    textlist=["cancer that begins in the lungs and most often occurs in people who smoke cancer", "A cancer that begins in the lungs and most often occurs in people lungs who smoke"]
    #words=extract_ngrams(text,1)
    #print(words)
    #print(len(words))

    pars=["cancer", "lungs", "people", "smoke"]
    count=0
    for i in textlist:
      lent=len(textlist)
      # training
      count+=1
      p1 = Agent("curater")
      words=p1.extract_ngrams(i,1)
      st = Environment(p1, i, words, pars)
      print("Iteration number...",count,"/",lent)
      st.play()
      #print(len(finalPARtag))
      st.finaltag()
      df=p1.updateStatetags(finalPARtag,finalBtag,finalAtag,finalrwd)
      ###df2=st.finaltag()
    print("trained model shape: ", df.shape)
    p1.savePolicy(df)
    print("RL model Training is completed ")



def loadPolicy(file):
    fr = open(file, 'rb')
    statevalue = pickle.load(fr)
    df=statevalue.values.tolist()
    return statevalue

statevalue=loadPolicy("/content/test1_curater")
print(statevalue)

# Function to generate n-grams from sentences.
def extract_ngrams(data, num):
    import nltk
    from nltk.util import ngrams
    nltk.download('punkt')
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]



def tags (text, par):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    from spacy.symbols import nsubj, VERB, NOUN
    doc = nlp(text.lower())
    for token in doc:
        if token.text == par.lower():
          if token.pos == NOUN:
            return 1
          elif token.pos == VERB:
            return 2
          else:
            return 3



def updateState(text, Ngramwords, position):
    import pandas as pd
    df2=pd.DataFrame()
    PARtag=[]
    Btag=[]
    Atag=[]
    B2tag=position-1
    A2tag=position+1
    Ncount= len(Ngramwords)
    value=Ncount-1
    #print("value", value)
    tagnumber=tags(text, Ngramwords[position])
    #print("tagnumber", tagnumber)
    PARtag.append(tagnumber)
    if A2tag <= value:
        Atagnumber=tags(text,Ngramwords[A2tag])
        Atag.append(Atagnumber)
    else:
        Atag.append(4)
    if B2tag >= 1:
        Btagnumber=tags(text,Ngramwords[B2tag])
        Btag.append(Btagnumber)
    else:
        Btag.append(5)
    df2=pd.DataFrame()
    df2["tag"]=PARtag
    df2["Btag"]=Btag
    df2["Atag"]=Atag
    return df2



text="A  that begins in the lungs and most often occurs in people who smoke cancer"

def Prediction(text):
  
  import numpy as np
  import pandas as pd
  import pickle  
  PAR=[]
  Ngram=extract_ngrams(text, 1)
  for i in range(len(Ngram)):
    statevalue=updateState(text, Ngram, i)
    oldStatevalue=loadPolicy("/content/test1_curater")
    #print(oldStatevalue)
    olddf=oldStatevalue.drop_duplicates()
    olddf2=olddf.filter(["tag", "Btag", "Atag"])
   # olddf3=olddf.filter(["tag", "Btag"])
    #print("olddf3", olddf3)
    #statevaluelimited=olddf.filter(["tag","Btag"])
   # print("statevaluelimited", statevaluelimited)
    df = pd.concat([olddf2, statevalue]) # concat dataframes  
    #df2 = pd.concat([olddf3, statevaluelimited]) # concat dataframes  
    #duplicateRowslimited=df2.duplicated(keep='last')
    duplicateRowsDF=df.duplicated(keep='last')    
    #print(duplicateRowsDF)
    listdupl=duplicateRowsDF.tolist()
    #listdupllimited=duplicateRowslimited.tolist()
    if True in listdupl:
      indexNumber=listdupl.index(True)
      row2 = olddf.iloc[indexNumber]
      if row2["Reward"]==1:
        PAR.append(Ngram[i])
    # elif True in listdupllimited:
    #   newindexNumber=listdupllimited.index(True)
    #   row3 = olddf.iloc[newindexNumber]
    #   if row3["Reward"]==1:
    #     PAR.append(Ngram[i])


  print("*********************Input text for Single Keyword Prediction*******************************")
  print("text :", text)
  print("Predicted Keywords are", PAR)
    #print(number)

dictvaluefinal=Prediction(text)





