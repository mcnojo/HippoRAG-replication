from string import Template


passages = [
'''
Luis Miguel Assuncao Joaquim (born 5 March 1979 in Vila Franca de Xira, Lisbon), known as Alhandra,
is a Portuguese retired footballer who played mainly as a left back -- he could also appear as a midfielder.
''',
'''
Vila Franca de Xira is a municipality in the Lisbon District in Portugal. The population in 2011 was
136,886, in an area of 318.19 km2. Situated on both banks of the Tagus River, 32 km north-east of the
Portuguese capital Lisbon, settlement in the area dates back to neolithic times, as evidenced by findings
in the Cave of Pedra Furada. Vila Franca de Xira is said to have been founded by French followers of
Portugal's first king, Afonso Henriques, around 1200.
''',
'''
Chirakkalkulam is a small residential area near Kannur town of Kannur District, Kerala state, South
India. Chirakkalkulam is located between Thayatheru and Kannur City. Chirakkalkulam's significance
arises from the birth of the historic Arakkal Kingdom.
''',
'''
The Frank T. and Polly Lewis House is located in Lodi, Wisconsin, United States. It was added to the
National Register of Historic Places in 2009. The house is located within the Portage Street Historic
District.
''',
'''
In the U.S., the issuance of birth certificates is a function of the Vital Records Office of the states, capital
district, territories and former territories
'''
]


# NER prompt: extract named entities from a passage
offline_ie_entities = Template('''
Instruction:
Your task is to extract named entities from the given paragraph.
Respond with a JSON list of entities.

One-Shot Demonstration:

Paragraph:
```
Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English
and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music
portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related
features.
```

Output:
{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi","English", "May 2008",
"PlanetRadiocity.com"]}


INPUT STARTS HERE:

Paragraph:

$_offline_passage
''')


# RDF triple extraction prompt: build (head, relation, tail) triples from passage + entity list
offline_ie_connections = Template('''
Instruction:
Your task is to construct an RDF (Resource Description Framework) graph from the given passages and
named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.
Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each
passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
Convert the paragraph into a JSON dict, it has a named entity list and a triple list.

One-Shot Demonstration:

INPUT:
Paragraph:
```
Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English
and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music
portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related
features.
```

{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi","English", "May 2008",
"PlanetRadiocity.com"]}

OUTPUT:
{"triples":
 [
 ["Radio City", "located in", "India"],
 ["Radio City", "is", "private FM radio station"],
 ["Radio City", "started on", "3 July 2001"],
 ["Radio City", "plays songs in", "Hindi"],
 ["Radio City", "plays songs in", "English"],
 ["Radio City", "forayed into", "New Media"],
 ["Radio City", "launched", "PlanetRadiocity.com"],
 ["PlanetRadiocity.com", "launched in", "May 2008"],
 ["PlanetRadiocity.com", "is", "music portal"],
 ["PlanetRadiocity.com", "offers", "news"],
 ["PlanetRadiocity.com", "offers", "videos"],
 ["PlanetRadiocity.com", "offers", "songs"]
 ]
}




INPUT STARTS HERE:

Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
$_paragraph_for_entities
```
$_entity_list

''')


# query-time NER prompt: extract entities relevant to answering a question
online_ie = Template('''
Instruction:
You're a very effective entity extraction system. Please extract all named entities that are important for
solving the questions below. Place the named entities in JSON format.

One-Shot Demonstration:

Question: Which magazine was started first Arthur's Magazine or First for Women?

Output:
{"named_entities": ["First for Women", "Arthur's Magazine"]}


INPUT STARTS HERE:

Question: $_online_question
''')
