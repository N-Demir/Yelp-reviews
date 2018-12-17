# Yelp-reviews
A CS229 final project by Nikita Demir, Jose Giron, and Jonathan Gomes Selman

There is a clear need with the rise of e-commerce to reliably detect fake reviews and misinformation online. We set out to do this using a variety of simple and complex machine learning algorithms and compared/contrasted them. 

We used a Yelp review dataset provided originally by Yelp but now hosted by [insert here] that contained 60,000 real and fake labeled Yelp restaurant reviews. We found surprisingly that the task of classification was extremely hard and that simply using word content was not a distinguishing enough feature. We then sought out to use behavorial features gotten from an accompanying metadata dataset for every review; however, these also were not very distinguishing.

Given the difficulty of classification, we then attempted to find out if we could somehow generate convincing fake restaurant reviews. Using a separate dataset of around 3 million restaurant reviews we trained a sequence to sequence model based on the OpenNMT framework to take an input context (review rating, type of food, location, etc.) and output a convincing review. Are results in this case were actually quite impressive and encouraging.

Future work on the matter should lead to even more fruitful results.

Our paper can be found as FinalReport.pdf
