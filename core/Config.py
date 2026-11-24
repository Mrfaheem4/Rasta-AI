from oel_instance import stamp

members = [
    ("Faheem Ahmed Khan", "534662"),
    ("Abdullah Waleed Ahmed", "509666"),
    ("Syed Ehtisham Haider Bukhari", "511754"),
    ("M Rehan Haider", "508160"),
]
cmsIds = [cms for _, cms in members]
stampLine = stamp(members)
cityBounds = {
    "south": 33.7073,
    "west": 73.0539,
    "north": 33.7307,
    "east": 73.0821,
    "name": "Islamabad F-6 / F-7 / Blue Area",
}
