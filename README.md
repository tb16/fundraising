# Fundraising
Many people suffer from unexpected problems in their life. When the situation becomes terrible to handle the financial problems, one would look for the sources such as government or charity organizations. The quicker/easier way of collecting funds would be starting the fundraising campaign through the crowdfunding website. In this project, I have analysed the people's story in the gofundme website, such as, what kind of common issues are there(which might help government and charity organization to design their policy) and what kind of word choices/sentiments help people to increase the fundraising amount.
  Success of fundraising campaign might depend on the choice of the words used to explain the problems and also the networking strength of the organizer of fundraising campaign. I have scraped different cases/stories of people in gofundme website, stored in MongoDB and processed in python. I did the exploratory analysis of the data to understand the statistics of the campaigns:
  1. In what category, people are seeking for help?
  2. What is the average target amount and how long does it take to reach the goal?
  3. Average number of social media share and likes to the total amount collected and average colection rate

For the modeling of the campaigns success(overfunded or underfunded), I took into account the following features:
i) Title
ii)Story, length of the story
iii)Number of facebook friends, number of shares
iv) Campaign duration
v)Target amount, collected account
vi) Ratio of collected amount/target amount, number of shares and
  number of fb friends to the number of contributors
vi) Location, and start-date
vii)Category

Several machine learning modeling eg, logistic regression, decision trees, random forest, SVM, kNN, NMF are used to explain the
outcome of the campaign.

  Natural Language Processing(NLP) on the story of each individual cases is used to identify the most prevalent issues, predict the similar success stories for lookup, and suggest the significant keywords.



# Running the model:
