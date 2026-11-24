from oel_instance import stamp

members = [
    ("Faheem Ahmed Khan", "534662"),
    ("Abdullah Waleed Ahmed", "509666"),
    ("Syed Ehtisham Haider Bukhari", "511754"),
    ("M Rehan Haider", "508160"),
]
cmsIds = [cms for _, cms in members]
stampLine = stamp(members)
