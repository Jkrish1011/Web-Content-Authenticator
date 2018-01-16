# Web-Content-Authenticator
This is a Firefox plugin which gives a confidence score to each of the links returned by the Google Search Engine.

<ul>
  <li><a href="#Intro">Introduction</a></li>
    <li><a href="Depth">In-Depth</a></li>
  </ul>
  
  <p id="Intro">This project is based on the fact of how much we can trust a link found over the Internet. This is a firefox plugin powered by a python Django framework in the backend which calculates a score for each link returned by the Google search engine after considering various factors.
  </p>
  
  <p id="Depth">
  <b>Factors Considered </b>
  The factors which are considered now is whether the links is using https or http protocol and the domain of the links.
  
  The domains in the list [org,edu etc] are given higher priorities if found in the links, then comes the government websites such as [in,uk, etc] followed by the commcercial websites.
  Then a correlation function is applied on these resultset to generate a score which is in between 0 and 1 . Higher the score, better trust worthy are the links.
  </p>
  
# Future developments
We have planned to consider various number of factors to contribute to give a more appropriate score. Such as number of web trackers working in the websites, number of out going links, Using sentimental analysis to get the motive of the content, do a back ground search of the author who wrote it etc. 


  
