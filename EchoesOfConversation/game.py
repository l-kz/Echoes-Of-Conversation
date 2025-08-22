import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pygame
from pygame.locals import *

# downloading all required nltk data:
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

contractions = {
    "that's": "that is",
    "there's": "there is",
    "aren't": "are not",
    "weren't": "were not",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "don't": "do not",
    "didn't": "did not",
    "doesn't": "does not",
    "let's": "let us",
    "hasn't": "has not",
    "hadn't": "had not",
    "haven't": "have not",
    "it's" : "it is",
    "isn't": "is not",
    "won't": "will not",
    "can't": "can not",
    "how's": "how is",
    "where's": "where is",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "you're": "you are",
    "they're": "they are",
    "they'd": "they would",
    "he's": "he is",
    "he'd": "he would",
    "he'll": "he will",
    "she's": "she is",
    "she'd": "she would",
    "she'll": "she will",
    "you'll": "you will",
    "you'd": "you would",
}

# function to chnage contractions such as "That's" to "that is"
def splitContractions(contraction):
    words = contraction.split()
    splitWords = [contractions[word.lower()] if word.lower() in contractions else word for word in words]
    return " ".join(splitWords) 

# preprocessing data by
# 1 handling contractions
# 2 removing number, punctuation, special characters
# 3 tokenizing the lowercase, filteres sentence
# 4 removing stopwords
# 5 lemmatizing tokens

def preprocessData(data):
    # splitting contractions 
    data = splitContractions(data)
    # removing any punctuation/ special characters/ numbers using regex, re
    # only keeping letters and spaces
    data = re.sub(r'[^a-zA-Z\s]', '', data)
    # tokenizing the filtered sentence using .split()
    filteredWords = data.split()
    # creating a set of stopwords using nltk library
    stopWords = set(stopwords.words('english'))
    # removing stopwords, and lowercasing words using .lower()
    filteredWords = [item for item in filteredWords if item.lower() not in stopWords]
    # lemmatizing the filtered words using nltk library import: WordNetLematizer()
    nltkLemmatizer = WordNetLemmatizer()
    lemmatizedWords = [nltkLemmatizer.lemmatize(item) for item in filteredWords]
    return ' '.join(lemmatizedWords)

correctAnswers = [
    [
        "I seek knowledge and adventure. I know the dangers, but I’m prepared, and I believe I have what it takes to overcome any challenge",
        "I am embarking on a learning journey for self-growth",
        "The thrill of adventure, I know the danger",
        "I need money, but I am also here to learn and further improve my skills",
        "I am a scholar, seeking to learn more about the world",
        "I have come as a test of strength",
        "I have come to enhance my skills and grow stronger",
        "I seek wisdom and experience that will guide me in future challenges",
        "My goal is self-improvement through knowledge and perseverance",
        "I’m here to test my limits and gain a deeper understanding of the world",
        "I pursue knowledge to help those who cannot help themselves",
        "I am a student, studying to become a doctor and I require more knowledge of more powerful potions",
        "I seek mastery in my craft and understanding of the mysteries beyond",
        "I seek to find out more about the world",
        "I seek a balance of wealth and knowledge, this will be my key to success",
        "I’m here to challenge myself and embark on an adventure",
        "I seek knowledge",
        "I seek adventure",
        "I seek the skills and wisdom that come only through facing great trials",
        "My journey is on of self-discovery, and I seek to become wiser in the process",
        "To become smarter and stronger",
        "To heighten my skills",
        "To gain strength, and become stronger and more powerful",
        "I desire to learn and grow, for knowledge is the ultimate power",
        "I seek the power to protect those I care about",
        "I seek understanding to unlock my potential and push beyond my limits",
        "My pursuit is of knowledge, to broaden my mind and sharpen my skills",
        "To protect my loved ones",
        "I seek experience that will allow me to stand stronger in the face of future challenges",
        "To learn more about this magical, vast, beautiful world",
        "For the greater good of mankind",
        "For the thrill of adventure and the knowledge that comes with it",
        "I pursue wisdom to bring strength and balance to myself, and those around me",
        "To study",
        "To learn more about the world",
        "To discover new wonders",
        "To study for the benefit of humanity",
        "To grow strong enough to protect my friends and family from danger",
        "To test how far I have come with my strength",
        "In search of new wonders of the world",
        "In search of medicine"

    ],
    [
        "more than enough", "studied everything i can", "studied every path", "Years’ worth", "lots", 
        "too much", "enough so that I feel very well prepares", "I am confident in the amount that I have trained", 
        "a huge amount", "a very long time", "I have dedicated years to training", "a large period of time", "a lot", 
        "too much", "countless hours", "as much as possible", "as much as I could", "a lifetime’s worth", 
        "every day", "every time I get", "all my time", "all day, every day", "an endless amount of time", 
        "a significant amount of time", "every waking moment", "years of dedication has been put into this", 
        "a significant amount", "too much to consider, I’ve lost track", "as much as possible", "day and night", 
        "countless days", "my whole life", "I have been fully committed", "with full dedication", "for years of my life", 
        "uncountable hours", "uncountable amount of time", "always", "from dawn till dusk", "from sunrise to sunset", "wholeheartedly", 
        "with full focus", "I have been completely devoted", "whenever I get the opportunity to", "whenever I can", "whenever possible", 
        "with passion and perseverance", "a substantial amount of effort has been put into it", "to the best of my ability", 
        "for as long as I can remember", "intensely and immensely", "around the clock", "all my time", "with full effort", 
        "with unwavering focus", "great discipline has been put into it", "endless dedication", "religiously", 
        "I am passionate, focused and persistent on my training", "countless weeks, months, years", "unyielding dedication", 
        "I feel very prepared", "Extremely", "I am very well prepared", "I have prepared my whole life for this", 
        "tonnes of time", "too much time", "I am prepared for any situation or scenario"

    ],
    [
        "for the greater good of mankind", "for the best", "to help me, my friends and my family", "for good", 
        "to aid those who need it", "to investigate and study it", "to bring peace and prosperity", 
        "to save those i care about", "to uplift my life", "to ensurea better future", "to donate to those in need", 
        "to study its medicinal purposes", "to enhance the well being of my community", 
        "for education purposes, and to bring great knowlege to others", "for protection, i need to protect my loved ones", 
        "for assist in science, and to complete my scientific research", "for the best", "for a good cause", 
        "to create well deserved opportunities", "for the betterment of society", "to make the world a better place", 
        "to create multiple opportunities", "to undertand this beautiful world better", "for the greater good of society", 
        "to ensure joy and happiness", "to bring good studies to the world", "to analyse in depth", "for research", 
        "to improve the quality of our lives", "to enhance my undertanding of the unknown", "to further develop my skills and knowledge",
          "keep it safe and away from danger", "to make the best use out of it", "for security"
    ],
    [
        "run away", "escape", "run", "dash and sprint", "run to save my life", "move away from it", "discretely escape", 
        "prioritise my life", "save my life", "do not fight", "do not attack", "refrain from startling the enemy", 
        "leave it alone", "leave", "repath and reposition my route", "find an alternative route", "flee and restreat", 
        "avoid it completely", "keep my distance from it", "i would hide and seek shelter", "I prioritise my safety", 
        "stay out of its way, and seek cover", "stay out of danger and harm's way", "outrun it or try to slip away", 
        "attempt to get away from it", "find a retreat", "i would avoid startling or provoking it", "find a safer route", 
        "i would be careful to make a stealthy escape", "reconsider my path", "avoid it at all costs", 
        "be safe, rather be safe than sorry", "navigate around it with caution", "avoid detection", 
        "i would reassess my journey", "allow it to pass", "wait carefully until it is safe", "proceed with caution", 
        "stay put until safe", "reconsider and relocate to a much safer route", "avoid it entirely", "leave it alone", 
        "i would not allow it to notice my presence", "avoid it seeing me"

    ],
    [
        "more than enough", "Years’ worth", "enough so that I feel very well prepares", 
        "I am confident in the amount that I have trained", "a huge amount", "a very long time", 
        "I have dedicated years to training", "a large period of time", "a lot", "too much", "countless hours", 
        "as much as possible", "as much as I could", "a lifetime’s worth", "every day", "every time I get", "all my time", 
        "all day, every day", "an endless amount of time", "a significant amount of time", "every waking moment", 
        "years of dedication has been put into this", "a significant amount", 
        "too much to consider, I’ve lost track", "as much as possible", "day and night", "countless days", 
        "my whole life", "I have been fully committed", "with full dedication", "for years of my life", "uncountable hours", 
        "uncountable amount of time", "always", "from dawn till dusk", "from sunrise to sunset", "wholeheartedly", 
        "with full focus", "I have been completely devoted", "whenever I get the opportunity to", "whenever I can", 
        "whenever possible", "with passion and perseverance", "a substantial amount of effort has been put into it", 
        "to the best of my ability", "for as long as I can remember", "intensely and immensely", "around the clock", 
        "all my time", "with full effort", "with unwavering focus", "great discipline has been put into it", "endless dedication", 
        "religiously", "I am passionate, focused and persistent on my training", "countless weeks, months, years", 
        "unyielding dedication", "I feel very prepared", "Extremely", "I am very well prepared", "I have prepared my whole life for this", 
        "tonnes of time", "too much time", "I am prepared for any situation or scenario"

    ]
]

incorrectAnswers = [
    [
        "I need money, I’m terrified, I don’t know what’s to come but I heard that there’s a lot of treasure there",
        "I’m only here for the treasure, nothing else matters",
        "I’m looking for some quick cash",
        "Does.t matter, let me pass through",
        "I came here for no reason, just curious",
        "I don’t care, neither should you",
        "Why do you care?",
        "I don’t need anything, I’m just bored, I have nothing to do and nothing to lose",
        "Take what I can and leave",
        "To cause chaos, and destroy",
        "To kill everyone and everything",
        "For revenge and to harm anyone who comes in my way",
        "To hurt those who hurt me",	
        "To find out how to ruin the world",
        "In order to destroy everything and everyone who has wronged me",
        "To find my enemy",
        "To die",
        "To kill or be killed",
        "To crush anyone who gets in my way",
        "this is stupid, horrible, bad, miserable",
        "Someone told me to",
        "Cos they said so",
        "Doesn’t matter, I have no goal, just going to wander around aimelessly",
        "it sounds fun",
        "I’m not sure, I came here on a whim",
        "I don’t know",
        "Money",
        "Not sure",
        "Shut up",
        "Why do you care?",
        "Who cares",
        "Nothing",
        "Dunno",
        "Only money",
        "Steal stuff",
        "Steal treasures",
        "why?",
        "I have come to gain power in order to rule the world",
        "Money, only money",
        "Haven’t decided yet",
        "Not sure yet",
        "What is there?",
        "To overthrow",
        "For fun",
        "None of your business"

    ],
    [
        "Nothing", "luck", "i might just get lucky who knows", "None", "I have not", 
        "this is stupid, horrible, bad, miserable, annoying", "little time", "Not much", 
        "a minuscule amount", "not enough", "never enough", "the bare minimum", "zero", "who cares?", 
        "I don’t remember", "barely any", "hardly any", "almost none", "nil", "Nada", "hardly anything", 
        "what does that mean?", "what’s that?", "why?", "why should i?", "I see no reason to", "that’s no your problem", 
        "a waste of time", "not yet", "haven’t started", "not interested", "why does it matter?", "does it matter?", 
        "forget it", "less than I should", "oh no", "less than you’d think", "I didn’t have time", "not enough time", 
        "I’m too busy for that", "why bother?", "training is not necessary, it is overrated", "not much effort has been put into it", 
        "seconds", "minutes", "I skipped that", "can’t remember", "I have no memory of that", "no", "I have not, and I will not", 
        "Zero effort", "I ignored that", "I’m strong enough, so no need", "no dedication", "negative", "a bit of joke", 
        "That’s not my priority", "enough, I guess", "I’ll worry about that later", "That’s not of your business", 
        "that’s pointless", "I don’t care", "I procrastinated that", "nah, I’m too lazy", "I don’t see the need to", 
        "Once or twice", "that does not apply to me", "I’m just going to wing it", "I have not prepared at all", 
        "no preparation", "zero preparation", "nope, nothing"
    ],
    [
        "none of your business", "money", "to sell", "to put it up for sale", "i do not know", "for someone else", 
        "it is not for me", "i need the money", "i want it all", "my greed and selfishness has led me here", 
        "to spread misinformation", "to rid the world of it", "i must destroy this world this must be stolen", 
        "not sure, it sounds fun", "it is a challenge, i guess", "no clue", "i'll worry about that later", 
        "who cares", "that is none of your business or concern", "that has nothing to do with you", 
        "i dont know, probably something stupid, dumb and irrational", "for bad and evil wrongdoings", 
        "i dont care about it", "i want it", "it is expensive, it will sell for a high price", 
        "it's quite popular, its in demand, there will be many buyers", "it is overrated, i will sell it off", "who knows", 
        "haven't decided yet", "why", "for revenge", "for evil", "to exploit its power", "to take advantage of its price", 
        "to become invicible, i will take over the world", "i need that luxury life", "i will show it off", 
        "i do whatever i want to do", "i do not owe you any explanation"
    ],
    [
        "nothing", "freeze", "fight", "battle it to the death", "face it", "stand still", "sit down", "attack", 
        "face my fears", "I do not know", "not sure", "no clue", "I wonder", "I wont worry about that", "I’m too strong", 
        "I’m the strongest", "I’m too powerful", "impossible for it to beat me", "why", "I’m not concerned", 
        "it won’t affect me", "fight it for sure", "it would be a worthy challenge", "endanger it before it kills me", 
        "attack it head on", "test my luck i guess", "confront it and stand my ground", 
        "charge at it in order to ambush it", "take it down of course", "i am much stronger", "there is no being stronger than me", 
        "it does scare me", "why would i fear it", "who would worry over such a thing", "fight it to the end", 
        "face it directly", "i would ignore it, i do not fear anything", "collide with it, it would be fun", 
        "oh a test of strength would be entertaining", "i would be tempted to recklessly charge into it", 
        "provoke it", "taunt the enemy", "it would be no big deal", "i might be careless and fight it", 
        "it would be a fun fight", "fight it aggressively", "I'll just wing it", "i dont know", 
        "laugh, it can't be stronger than me", "shrug it off", "it would be in my way", "get rid of it", 
        "kill it", "murder is the only solution", "violence is alwasy the answer", "i woul overpower it easily", 
        "why would i be scared of something so weak", "test myself, why not", "head straight towards it", 
        "keep going straight, all the enemies are weak"
    ],
    [
        "Nothing", "None", "I have not", "this is stupid, horrible, bad, miserable, annoying", "little time", 
        "Not much", " a minuscule amount", "not enough", "never enough", "the bare minimum", "zero", "who cares?", 
        "I don’t remember", "barely any", "hardly any", "almost none", "nil", "Nada", "hardly anything", 
        "what does that mean?", "what’s that?", "why?", "why should i?", "I see no reason to", 
        "that’s no your problem", "a waste of time", "not yet", "haven’t started", "not interested", 
        "why does it matter?", "does it matter?", "forget it", "less than I should", "oh no", "less than you’d think", 
        "I didn’t have time", "not enough time", "I’m too busy for that", "why bother?", 
        "training is not necessary, it is overrated", "not much effort has been put into it", "seconds", "minutes", 
        "I skipped that", "can’t remember", "I have no memory of that", "no", "I have not, and I will not", "Zero effort", 
        "I ignored that", "I’m strong enough, so no need", "no dedication", "negative", "a bit of joke", "That’s not my priority", 
        "enough, I guess", "I’ll worry about that later", "That’s not of your business", "that’s pointless", "I don’t care", 
        "I procrastinated that", "nah, I’m too lazy", "I don’t see the need to", "Once or twice", "that does not apply to me", 
        "I’m just going to wing it", "I have not prepared at all", "no preparation", "zero preparation", "nope, nothing"
    ]
]

# NLP 
naiveBayesClassifier = MultinomialNB()
vectorizer = CountVectorizer()

pygame.init()
time = pygame.time.Clock()
# setting frames per second
fps = 60

# setting size of screen of game
widthOfScreen = 750
heightOfScreen = 750

# creating display window for game - initially blank
screenDisplay = pygame.display.set_mode((widthOfScreen, heightOfScreen))

# creating a name (caption) for display window
pygame.display.set_caption('Echoes Of Conversation')

# defining main game variables
# set tile size to one twentieth (750/20) of screen
sizeOfTile = 37.5
# game over
gameOver = 0
# start/end menu
menu = True
gameLevel = 0


# creating images on game display
# loading background image
backgroundImage = pygame.image.load('image/blackBackground.png')
# resizing background image to fit 1000x1000 display
backgroundImage = pygame.transform.scale(backgroundImage, (widthOfScreen, heightOfScreen))
# loading restart image
restartImage = pygame.image.load('image/restart.png')
# resizing restart image , initial size: 42x11
restartImage = pygame.transform.scale(restartImage, (294,77))
# loading start image
startImage = pygame.image.load('image/start.png')
# reszing, initally, 30x11
startImage = pygame.transform.scale(startImage, (150,55))
# loading end image
endImage = pygame.image.load('image/end.png')
#reszing, initally 20x11
endImage = pygame.transform.scale(endImage, (100, 55))
endAtDeathImage = pygame.transform.scale(endImage, (140, 77))
# title image
titleImage = pygame.image.load('image/echoesOfConversation2.png')
# resizing title image, og size 131x13
titleImage = pygame.transform.scale(titleImage, (655,65))
# player on menu
menuPlayer = pygame.image.load('image/player1Stand.png')
# resizing player
menuPlayer = pygame.transform.scale(menuPlayer, (50,75))
# gatekeeper on menu
menuGatekeeper = pygame.image.load('image/gatekeeper.png')
# resizing gatekeeper
menuGatekeeper = pygame.transform.scale(menuGatekeeper, (50,75))
# last level completion congrats image
congratsEnd = pygame.image.load('image/congrats.png')
# resizing congrats image
congratsEnd = pygame.transform.scale(congratsEnd, (665, 77))
# main menu button upon completion of last level
menuButton = pygame.image.load('image/mainMenu.png')
# resizing menu button image
menuButton = pygame.transform.scale(menuButton, (336, 77))
# question images
question1Image = pygame.image.load('image/question1.png')
question1Image = pygame.transform.scale(question1Image, (650, 115))
question2Image = pygame.image.load('image/question2.png')
question2Image = pygame.transform.scale(question2Image, (744, 54))
question3Image = pygame.image.load('image/question3.png')
question3Image = pygame.transform.scale(question3Image, (733, 53))
question4Image = pygame.image.load('image/question4_2.png')
question4Image = pygame.transform.scale(question4Image, (741, 83))
question5Image = pygame.image.load('image/question5.png')
question5Image = pygame.transform.scale(question5Image, (711, 53))


class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.mouseClicked = False

    def draw(self):

        # for any action on mouse taken
        actionTaken = False
        # getting the position of the mouse cursor using pygames get_pos()
        position = pygame.mouse.get_pos()
        if self.rect.collidepoint(position):
            # if button is beingclicked (left click only- index 0), and if mouse button hasnt already been clicked
            if pygame.mouse.get_pressed()[0] == 1 and self.mouseClicked == False:
                actionTaken = True
                self.mouseClicked = True

        if pygame.mouse.get_pressed()[0] == 0:
            self.mouseClicked = False

        # drawing the buytton
        screenDisplay.blit(self.image, self.rect)

        return actionTaken
# creating a class for the player
class Player():
    def __init__(self, x, y):
        self.resetPlayer(x, y)

    def updatePlayer(self, gameOver):
        changeInX = 0
        changeInY = 0
        animationRate = 3

        if gameOver ==0:

            # moving player with key presses
            key = pygame.key.get_pressed()
            # if left key is pressed
            if key[pygame.K_LEFT]:
                changeInX -= 4.9
                self.count += 1
                self.direction = -1
            if key[pygame.K_RIGHT]:
                changeInX += 4.9
                self.count += 1
                self.direction = 1
            if key[pygame.K_SPACE] and self.jumped == False and self.floating == False:
                self.velocityOfY = -15.4
                self.jumped = True
            if key[pygame.K_SPACE] == False:
                self.jumped = False
            if key[pygame.K_UP] and self.jumped == False and self.floating == False:
                self.velocityOfY = -15.4
                self.jumped = True
            if key[pygame.K_UP] == False:
                self.jumped = False
            # ressting character animation for when player is stationary
            if key[pygame.K_LEFT] == False and key[pygame.K_RIGHT] == False:
                self.count = 0
                self.index = 0
                if self.direction ==1:
                    self.image = self.playerMovesRight[self.index]
                if self.direction ==-1:
                    self.image = self.playerMovesLeft[self.index]

            # animate character
            if self.count > animationRate:
                self.count = 0
                self.index += 1
                if self.index >= len(self.playerMovesRight):
                    self.index = 0
                if self.direction ==1:
                    self.image = self.playerMovesRight[self.index]
                if self.direction ==-1:
                    self.image = self.playerMovesLeft[self.index]
                


            # adding gravity to return player to platform
            self.velocityOfY += 1.40
            if self.velocityOfY > 15:
                self.velocityOfY = 15
            changeInY = self.velocityOfY


            #  collision of platforms/ objcts
            self.floating = True
            for tile in world.listOfTiles:
                # checking if player collides with a platform to the left/right
                if tile[1].colliderect(self.rect.x + changeInX, self.rect.y, self.width, self.height):
                    changeInX = 0

                # checking if player collides in y direction (up, downm)
                # using colliderect which is a built in collision tool in the pygame library
                if tile[1].colliderect(self.rect.x, self.rect.y + changeInY, self.width, self.height):
                    # keeping player above ground (jumping)
                    if self.velocityOfY < 0:
                        changeInY = tile[1].bottom - self.rect.top
                    elif self.velocityOfY >= 0:
                        changeInY = tile[1].top - self.rect.bottom
                        # once collided with a higher platform, make velocity 0, (bounce off effect)
                        self.velocityOfY = 0
                        # when on a pltform, player caN JHUMP AGAIN
                        self.floating = False

            # chekcing for collisions with enemies/ lava
            if pygame.sprite.spritecollide(self, groupOfBarnacles, False):
                gameOver = -1
            if pygame.sprite.spritecollide(self, groupOfLava, False):
                gameOver = -1
                print("game over")
            if pygame.sprite.spritecollide(self, groupOfKeys, False):
                gameOver = 1



            # update player's position
            self.rect.x += changeInX
            self.rect.y += changeInY

              

        elif gameOver == -1:
            self.image = self.deadGameOver
            # for player to float out of screen when dead 
            #if self.rect.y> 300:
            self.rect.y -=3
        # draw player
        screenDisplay.blit(self.image, self.rect)
        # creating a visible rectangle for player, white, 1 pixel, around player
        # pygame.draw.rect(screenDisplay, (255, 255, 255), self.rect, 1)

        return gameOver
    
    def resetPlayer(self, x, y):
        self.playerMovesRight = []
        self.playerMovesLeft = []
        self.index = 0
        # counter
        self.count = 0
        # using a for loop to add all images to construct walking animation
        for i in range(1, 12):
            imageMovesRight = pygame.image.load(f'image/p1Walk{i}.png')
            imageMovesLeft = pygame.image.load(f'image/p1WalkL{i}.png')
            # resizing image of player
            imageMovesRight = pygame.transform.scale(imageMovesRight, (35, 60))
            imageMovesLeft = pygame.transform.scale(imageMovesLeft, (35, 60))
            # addign images to list
            self.playerMovesRight.append(imageMovesRight)
            self.playerMovesLeft.append(imageMovesLeft)
        self.deadGameOver = pygame.image.load('image/slime_squashed.png')
        # first image is at index 0, therefore, loads first image
        self.image = self.playerMovesRight[self.index]    
        # creating rectangle for player in order to get (x, y)
        self.rect = self.image.get_rect()
        # getting x, y coordinates
        self.rect.x = x
        self.rect.y = y
        # getting width and height usin built in get_width() and get_height()
        self.height = self.image.get_height()
        self.width = self.image.get_width()
        self.velocityOfY = 0
        self.jumped = False
        self.direction = 0
        self.floating = True
    





# creating a class called gridWorld
# naming gridWorld as it is a world constructing of grids
class GridWorld():
    # data in the constructor is the grid world data that will be loaded onto it
    def __init__(self, data):
        # creating an empty list to add tiles to
        self.listOfTiles = []
        # loading images onto grid world
        stoneImage = pygame.image.load('image/stoneCentre.png')
        steppingStoneImage = pygame.image.load('image/stone.png')

        rowCounter = 0
        # using a for loop to run through each row in grid world data, and make anything with number '1' a stone
        for row in data:
            columnCounter = 0
            for area in row:
                if area == 1:
                    # changing size of tile to fit area
                    image = pygame.transform.scale(stoneImage, (sizeOfTile, sizeOfTile))
                    # creating a rectangle from image
                    imageRect = image.get_rect()
                    imageRect.x = columnCounter * sizeOfTile
                    imageRect.y = rowCounter * sizeOfTile
                    # creating a tuple named tile to store tile
                    tile = (image, imageRect)
                    # add tile to list of tiles
                    self.listOfTiles.append(tile)
                if area == 2:
                    # changing size of tile to fit area
                    image = pygame.transform.scale(steppingStoneImage, (sizeOfTile, sizeOfTile))
                    # creatign a rectangle from image
                    imageRect = image.get_rect()
                    imageRect.x = columnCounter * sizeOfTile
                    imageRect.y = rowCounter * sizeOfTile
                    # creatign a tuple named tile to store tile
                    tile = (image, imageRect)
                    # add tile to list of tiles
                    self.listOfTiles.append(tile)
                if area == 3:
                    # creating an instance of barnacle
                    barnacle = Barnacle(columnCounter * sizeOfTile, rowCounter * sizeOfTile + 8)
                    # addign barnacle to group
                    groupOfBarnacles.add(barnacle)
                if area == 4:
                    lava = Lava(columnCounter * sizeOfTile, rowCounter * sizeOfTile)
                    groupOfLava.add(lava)
                if area == 5:
                    key = Key(columnCounter * sizeOfTile, rowCounter * sizeOfTile)
                    groupOfKeys.add(key)


                # iterate and add to column counter
                columnCounter += 1
            rowCounter += 1

    # method to draw tiles on gird
    def draw(self):
        for tile in self.listOfTiles:
            # each tile is svaed as a tuple
            # first tile is the imgae, second tile is rectanglw
            screenDisplay.blit(tile[0], tile[1])
            pygame.draw.rect(screenDisplay, (255, 255, 255), tile[1], 1)

class NLP():
    def askQuestion(self, questionImage, level):
        # receiving data lists for suitable question (5 in total)
        correctAnswersLevel = correctAnswers[level]
        incorrectAnswersLevel = incorrectAnswers[level]
        # putting data together in one list
        trainData = correctAnswersLevel + incorrectAnswersLevel
        # assigning numerical value 1 for correct, and 0 for incorrect
        trainLabels = [1] * len(correctAnswersLevel) + [0] * len(incorrectAnswersLevel)
        # preprocessing the data
        preprocessedData = [preprocessData(data) for data in trainData]
        # vectorizing data 
        trainVectors = vectorizer.fit_transform(preprocessedData)
        # training the naive bayes classifier
        naiveBayesClassifier.fit(trainVectors, trainLabels)
        # font of the text
        screenDisplay.blit(questionImage, (10,150))
        screenDisplay.blit(menuGatekeeper, (40, 280))
        screenDisplay.blit(menuPlayer, (40, 450))
        font = pygame.font.Font(None, 36)
        pygame.display.update()
        playerAnswer = ""
        typing = True
        while typing:
            for event in pygame.event.get():
                # ending game if window is closed
                if event.type == pygame.QUIT:
                    typing = False
                    pygame.quit()

                # logic for key presses
                if event.type == pygame.KEYDOWN:
                    # player submits answer by enter button
                    if event.key == pygame.K_RETURN:
                        preprocessedAnswer = preprocessData(playerAnswer)
                        answerVector = vectorizer.transform([preprocessedAnswer])
                        calculation = naiveBayesClassifier.predict(answerVector)
                        
                        if calculation[0] == 1:
                            typing = False
                            pygame.event.clear()
                            return True 
                        # clears the text box
                        else: 
                            playerAnswer = ""
                    # deletign characters
                    elif event.key == pygame.K_BACKSPACE:
                        playerAnswer = playerAnswer[:-1]
                    # entering charcaters
                    else:
                        playerAnswer += event.unicode
            # rgb value for purple = 111 49 152 6f3198
            pygame.draw.rect(screenDisplay, (111, 49, 152), (120, 450, 500, 100))
            displayAnswer = font.render(playerAnswer, True, (255, 255, 255))
            
            screenDisplay.blit(displayAnswer, (130, 475))
            pygame.display.update()
        return False




# addign barnacles as a potential enemy
# barnacle class to be a child of the pygame sprite class

class Barnacle(pygame.sprite.Sprite):
    def __init__(self, x, y):
        # initialise the barnacle with the sprite features
        pygame.sprite.Sprite.__init__(self)
        # loadign image of barnacle
        self.image = pygame.image.load('image/barnacle.png')
        # resizing image
        self.image = pygame.transform.scale(self.image, (20, 30))
        # creatiung a rectanbgle for barnacle
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.move = 1
        # crating a count for moves
        self.countOfMoves = 0

    # making the barnalces move left and right to increase diffficulty
    def update(self):
        self.rect.x += self.move
        self.countOfMoves += 0.75
        # swapping directions after a couple steps
        # absolute values to counter=attack the negative values
        if abs(self.countOfMoves) > 22:
            self.move *= -1
            # to move the enemy to the right (from centre) then to the left
            self.countOfMoves *= -1


# creating a lava/slime class
class Lava(pygame.sprite.Sprite):
    def __init__(self, x, y):
        # initialise the lava with the sprite features
        pygame.sprite.Sprite.__init__(self)
        # loadign image of lava
        imageOfLava = pygame.image.load('image/lavaMid.png')
        # resizing image
        self.image = pygame.transform.scale(imageOfLava, (sizeOfTile, sizeOfTile))
        # creatiung a rectanbgle for lava
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Key(pygame.sprite.Sprite):
    def __init__(self, x, y):
        # initialise the lava with the sprite features
        pygame.sprite.Sprite.__init__(self)
        # loadign image of lava
        imageOfLava = pygame.image.load('image/key.png')
        # resizing image
        self.image = pygame.transform.scale(imageOfLava, (sizeOfTile , sizeOfTile))
        # creatiung a rectanbgle for key
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y



# list to store grid world data


gridWorldData1 = [
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 1, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1],
[1, 0, 0, 0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

gridWorldData2 = [
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 2],
[1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 1, 1, 1], 
[1, 2, 2, 2, 2, 1, 1, 2, 2, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1]
]

gridWorldData3 = [
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 0, 0, 0, 0, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

gridWorldData4 = [
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 2, 1], 
[1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 3, 0, 2, 4, 2, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 1], 
[1, 0, 0, 0, 0, 0, 2, 2, 1, 4, 4, 4, 4, 4, 1, 4, 4, 4, 1, 1], 
[1, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

gridWorldData5 = [
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 2, 2, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 5, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1], 
[1, 2, 0, 0, 2, 2, 2, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 2, 4, 1],
[1, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 0, 0, 0, 0, 2, 2, 2, 4, 4, 4, 4, 4, 2, 4, 4, 4, 1, 1, 1],
[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 2, 2, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# creating an instance of the player and world
# y of heightOfdScreen - 97.5 as 37.5 + 60 (height of player)
# player is put on top of one of the tiles
player = Player(80, heightOfScreen - 97.5)

# creating groups for non player/tile charcaters/ objects
groupOfBarnacles = pygame.sprite.Group()
groupOfLava = pygame.sprite.Group()
groupOfKeys = pygame.sprite.Group()

# creating list of game levels
gameLevels = [gridWorldData1, gridWorldData2, gridWorldData3, gridWorldData4, gridWorldData5]

# creating a list for allkquestions
questions = [question1Image, question2Image, question3Image, question4Image, question5Image]
# loading world from levels
worldList = gameLevels[gameLevel]
world = GridWorld(worldList)
# test
# world = GridWorld(gridWorldData5)

#when dead, restarting the game
restart = Button(widthOfScreen //2 - 275, heightOfScreen //2 - 50, restartImage)
start = Button(widthOfScreen // 2 - 200, heightOfScreen // 2 + 50, startImage)
end = Button(widthOfScreen //2 + 75, heightOfScreen // 2 + 50, endImage)
title = Button(widthOfScreen //2 - 325, heightOfScreen // 2 -200, titleImage)
endAtDeath = Button(widthOfScreen //2 + 105, heightOfScreen // 2 - 50, endAtDeathImage)
titlePlayer = Button(widthOfScreen // 2 - 155, heightOfScreen // 2 + 150, menuPlayer)
titleGatekeeper = Button(widthOfScreen //2 + 95, heightOfScreen // 2 + 150, menuGatekeeper)
congrats = Button(widthOfScreen //2 - 332, heightOfScreen // 2 -142, congratsEnd)
mainMenu = Button(widthOfScreen //2 - 167, heightOfScreen // 2 , menuButton)

# (without loop) so far game runs once (for a split second)
# a loop is now required
# running game
run = True
displayQuestion = False
levelStarted = False
currentQuestion = 0

while run:
    # using frame rate to limit how quickly code runs
    time.tick(fps)
    # displaying background
    screenDisplay.blit(backgroundImage, (0,0))
    # displaying start and end buttons when menu = True
    if menu == True:
        title.draw()
        titlePlayer.draw()
        titleGatekeeper.draw()
        if end.draw() == True:
            run = False

        if start.draw() == True:
            menu = False
            displayQuestion = True
    elif displayQuestion:
        nlp = NLP()
        if nlp.askQuestion(questions[currentQuestion], gameLevel):
            displayQuestion = False
            levelStarted = True
            currentQuestion += 1
        else:
            pass

    elif levelStarted:
        # drawing the world and grid
        world.draw()

        if gameOver == 0:
            # updatign entire group of barnalces
            # when player dies, barnacle sop moving
            groupOfBarnacles.update()
        # drawing barnacles on screen
        groupOfBarnacles.draw(screenDisplay)
        groupOfLava.draw(screenDisplay)
        groupOfKeys.draw(screenDisplay)
        # if player runs into lava/ enemy game over
        gameOver = player.updatePlayer(gameOver)


        # drawing the grid
        # grid()


        # of the player is dead
        if gameOver == -1:
            if restart.draw():
                player.resetPlayer(70, heightOfScreen - 97.5)
                gameOver = 0
            if endAtDeath.draw():
                
                menu = True
                player.resetPlayer(70, heightOfScreen - 97.5)
                gameOver = 0

        # if player touches key to complete level

        if gameOver == 1:
            gameLevel += 1
            if gameLevel <= 4:
                worldList = []
                # worldReset(gameLevel)
                # orrrr big or 
                player.resetPlayer(70, heightOfScreen - 97.5)
                groupOfBarnacles.empty()
                groupOfLava.empty()
                groupOfKeys.empty()
                worldList = gameLevels[gameLevel]
                world = GridWorld(worldList)
                gameOver = 0
                displayQuestion = True
                levelStarted = False
            

            else:
                congrats.draw()
                if mainMenu.draw():
                
                    
                    gameLevel = 0
                    
                    groupOfBarnacles.empty()
                    groupOfLava.empty()
                    groupOfKeys.empty()
                    worldList = gameLevels[gameLevel]
                    world = GridWorld(worldList)
                    
                    
                    
                    menu = True
                    player.resetPlayer(70, heightOfScreen - 97.5)
                    gameOver = 0
                



    # add a way to exit game
    for event in pygame.event.get():
        # x button in pygame code is denoted by pygame.QUIT
        if event.type == pygame.QUIT:
            # changing boolean run to false to exit loop
            run = False
    # updating display window with instructions
    pygame.display.update()
pygame.quit()

# testing the preprocessing steps
data_1 = "I have to enhance my skills and grow stronger"
data_2 = "My journey is on self-discovery, and I seek to become wiser in the process"
data_3 = "I’m not sure, I came here randomly"
data_4 = "I don't care, go away"
data_5 = "I love a good adventure, the thrill is what drives me"

preprocessed_1 = preprocessData(data_1)
preprocessed_2 = preprocessData(data_2)
preprocessed_3 = preprocessData(data_3)
preprocessed_4 = preprocessData(data_4)
preprocessed_5 = preprocessData(data_5)

print(preprocessed_1)
print(preprocessed_2)
print(preprocessed_3)
print(preprocessed_4)
print(preprocessed_5)

