import discord
import pandas as pd

from discord.ext import commands
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Training logistic model

df = pd.read_csv("microaggression_corpus.csv", header=0)

def remove_punctuation(s):
    removable_symbols = ["\'"]
    for symbol in removable_symbols:
        s = s.replace(symbol, "")
    return s

def list_to_str(lst):
    final_s = ""
    for s in lst:
        final_s += (s.lower() + " ")
    return final_s

df["Statement"] = df["Statement"].apply(remove_punctuation)

tokenizer = RegexpTokenizer(r"\w+")

df["Statement"] = df["Statement"].apply(tokenizer.tokenize)

df["Statement"] = df["Statement"].apply(list_to_str)

X_train = df.Statement
y_train = df.Category

logistic_regression = Pipeline(
    [("vect", CountVectorizer()),
    ("tfidf", TfidfTransformer()),
    ("clf",LogisticRegression(C=1000))]
)

logistic_regression.fit(X_train, y_train)

X_test = [
    "he looks so gay",
    "color doesnt exist",
    "where do you REALLY come from",
    "millennials dont know anything about technology",
    "i am going home after school",
    "if you're from spain then why dont u speak spanish?"
]

y_test = [
    "Heteronormativity",
    "Color blindness",
    "Treating someone like a perpetual foreigner",
    "Ageism",
    "Not a microaggression",
    "Assumptions about somone's background"
]

y_pred = logistic_regression.predict(X_test)

category_labels = [
    "Treating someone like a perpetual foreigner",
    "Color blindess",
    "Heteronormativity",
    "Assumptions about someone's background",
    "Ageism",
    "Not a microaggression"
]

print("accuracy %s" % accuracy_score(y_pred, y_test))

# Types of microaggressions covered by this bot

microaggression_codex = {
    "Treating someone like a perpetual foreigner":
        """This type of microaggression sends the message that a person is a foreigner in their own country based on their appearance, name, or aspects of their ethnic and racial identity.
        Example: Asking a person 'Where are you REALLY from?' implies that they don't belong in this country.
        """,
    "Color blindness":
        """These microaggressions deny the significance of the racial and ethnic experiences of BIPOC.
        Example: Saying 'I don't believe in race.'
        """,
    "Heteronormativity":
        """These microaggressions demean and exclude people who are not heterosexual.
        Example: Asking 'Who is the man in the relationship?', which perpetuates the exclusionary belief that heterosexuality is the norm.
        """,
    "Assumptions about someone's background":
        """These microaggressions make harmful and often untrue assumptions about an individual based on their racial, cultural, or ethnic background.
        Example: Asking 'Why are you bad at math? Aren't you Asian?' makes someone feel inferior for not fulfilling an unrealistic stereotype.
        """,
    "Ageism":
        """These microaggressions perpetuate harmful stereotypes about an individual based on their age.
        Example: Saying 'Old people are incompetent at using technology' is a blanket statement that dismisses the capability of senior citizens.
        """
}

# Bot code

description = "A bot that responds to and raises awareness about microaggressions."
token = open("TOKEN.txt","r").readline()

intents = discord.Intents.default()
intents.members = True

bot = commands.Bot(command_prefix="-", description=description, intents=intents)

@bot.event
async def on_ready():
    """Bot getting ready."""
    print(f"{bot.user} is ready!")

@bot.command()
async def greetings(ctx):
    """Greets the user."""
    person = ctx.author
    greetings_embed = discord.Embed(
        title=f"Hello, {person.display_name}!",
        description="""I'm Radar, a bot that aims to raise awareness about microaggressions.
        Type -learn to start learning.
         """,
         color=0xC9a0ff
         )
    await ctx.send(embed=greetings_embed)

@bot.command()
async def learn(ctx):
    """Begins the learning module."""

    initial_msg = discord.Embed(
        title="Learn About Microaggressions",
        description="""
        Microaggression is a term describing small acts of prejudice that take a huge toll on victims over time. A microaggression can be deliberate and conscious, or can be outside of one's conscious behavior.
        React to learn more about three forms of microaggressions.
         1️⃣ - Microassaults
         2️⃣ - Microinsults
         3️⃣ - Microinvalidations
         """,
         color=0xC9a0ff
        )

    initial_msg.set_footer(text=f"Information requested by: {ctx.author.display_name}")

    await ctx.send(embed=initial_msg)

    one = "1️⃣"
    two = "2️⃣"
    three = "3️⃣"

    def check_initial(reaction, user):
        return user == ctx.author and str(reaction.emoji) in [one, two, three]

    reaction, user = await bot.wait_for("reaction_add", check=check_initial)

    if str(reaction.emoji) == one:
        one_embed = discord.Embed(
            title="Microassaults",
            description="""
            Microassaults are a form of overt discrimination or criticism that is done to discredit a marginalized group.
            Examples:
                1) Using a racist slur in conversation
                2) Posting a historically offensive symbol
            """,
            color=0xC9a0ff
            )
        await ctx.send(embed=one_embed)
    elif str(reaction.emoji) == two:
        two_embed = discord.Embed(
            title="Microinsults",
            description="""
            Microinsults are communications that subtly convey a lack of respect for a demographic group.
            Examples:
                1) Saying that another person "speaks English well", implying that the person is a perpetual foreigner and foreigners are not expected to be fluent in English
                2) Seating a BIPOC couple in a more undesirable location than a white couple in a restaurant, indicating that they are second-class citizens
            """,
            color=0xC9a0ff
            )
        await ctx.send(embed=two_embed)
    elif str(reaction.emoji) == three:
        three_embed = discord.Embed(
            title="Microinvalidations",
            description="""
            A microinvalidation assails the identity and self-esteem of members of marginalized groups.
            Those who commit microinvalidations often dismiss and discredit an individual's experiences because they have not experienced something similar.
            Examples:
                1) Declaring "I don't see color", which denies a person's racial identity
                2) Invalidating a victim's feelings by telling them that they are being "petty" or "sensitive"
            """,
            color=0xC9a0ff
            )

        right_arrow = "➡️"
        three_embed.set_footer(text=f"Press {right_arrow} for more resources!")

        await ctx.send(embed=three_embed)

        def check_more_resources(reaction, user):
            return user == ctx.author and str(reaction.emoji) in [right_arrow]

        reaction, user = await bot.wait_for("reaction_add", check=check_more_resources)

        if str(reaction.emoji) == right_arrow:
            more_resources_embed = discord.Embed(
                title="Interested in learning more? Wonderful!",
                description="""
                Be sure to check out [Tiffany Alvoid's TED talk on eliminating microaggressions](https://www.youtube.com/watch?v=cPqVit6TJjw).
                You should definitely take a look at the University of Illinois Urbana-Champaign's [guide to responding to microaggressions](https://wie.engineering.illinois.edu/a-guide-to-responding-to-microaggressions/).
                """,
                color=0xC9a0ff
                )
            await ctx.send(embed=more_resources_embed)


@bot.event
async def on_message(message):
    """Filters channel messages based on type of microaggression (Not a microaggression is also a category)."""

    if message.author == bot.user:
        return

    elif len(message.content) > 15:
        message_category = logistic_regression.predict([message.content])
        if message_category[0] == "Not a microaggression":
            return
        else:
            microaggression_embed = discord.Embed(
                title=f"HALT! You have committed a microaggression related to {message_category[0].lower()}.",
                description=microaggression_codex[message_category[0]],
                color=0xFF0000
                )
            await message.channel.send(embed=microaggression_embed)

    else:
        await bot.process_commands(message)

    @bot.command()
    async def depart(ctx):
        """Bot going offline."""
        await bot.close()

bot.run(token)
