#!/usr/bin/env python

contextsid = {
    'country' : {'afghanistan': 0, 'albania': 1, 'algeria': 2, 'andorra': 3, 'angola': 4, 'antigua & deps': 5, 'argentina': 6, 'armenia': 7, 'australia': 8, 'austria': 9, 'azerbaijan': 10, 'bahamas': 11, 'bahrain': 12, 'bangladesh': 13, 'barbados': 14, 'belarus': 15, 'belgium': 16, 'belize': 17, 'benin': 18, 'bhutan': 19, 'bolivia': 20, 'bosnia herzegovina': 21, 'botswana': 22, 'brazil': 23, 'brunei': 24, 'bulgaria': 25, 'burkina': 26, 'burundi': 27, 'cambodia': 28, 'cameroon': 29, 'canada': 30, 'cape verde': 31, 'central african rep': 32, 'chad': 33, 'chile': 34, 'china': 35, 'colombia': 36, 'comoros': 37, 'congo': 38, 'costa rica': 39, 'croatia': 40, 'cuba': 41, 'cyprus': 42, 'czech republic': 43, 'denmark': 44, 'djibouti': 45, 'dominica': 46, 'dominican republic': 47, 'east timor': 48, 'ecuador': 49, 'egypt': 50, 'el salvador': 51, 'equatorial guinea': 52, 'eritrea': 53, 'estonia': 54, 'ethiopia': 55, 'fiji': 56, 'finland': 57, 'france': 58, 'gabon': 59, 'gambia': 60, 'germany': 61, 'ghana': 62, 'greece': 63, 'grenada': 64, 'guatemala': 65, 'guinea': 66, 'guinea-bissau': 67, 'guyana': 68, 'haiti': 69, 'honduras': 70, 'hungary': 71, 'iceland': 72, 'india': 73, 'indonesia': 74, 'iran': 75, 'iraq': 76, 'ireland': 77, 'israel': 78, 'italy': 79, 'ivory coast': 80, 'jamaica': 81, 'japan': 82, 'jordan': 83, 'kazakhstan': 84, 'kenya': 85, 'kiribati': 86, 'kosovo': 87, 'kuwait': 88, 'kyrgyzstan': 89, 'laos': 90, 'latvia': 91, 'lebanon': 92, 'lesotho': 93, 'liberia': 94, 'libya': 95, 'liechtenstein': 96, 'lithuania': 97, 'luxembourg': 98, 'macedonia': 99, 'madagascar': 100, 'malawi': 101, 'malaysia': 102, 'maldives': 103, 'mali': 104, 'malta': 105, 'marshall islands': 106, 'mauritania': 107, 'mauritius': 108, 'mexico': 109, 'micronesia': 110, 'moldova': 111, 'monaco': 112, 'mongolia': 113, 'montenegro': 114, 'morocco': 115, 'mozambique': 116, 'myanmar': 117, 'namibia': 118, 'nauru': 119, 'nepal': 120, 'netherlands': 121, 'new zealand': 122, 'nicaragua': 123, 'niger': 124, 'nigeria': 125, 'north korea': 126, 'norway': 127, 'oman': 128, 'pakistan': 129, 'palau': 130, 'panama': 131, 'papua new guinea': 132, 'paraguay': 133, 'peru': 134, 'philippines': 135, 'poland': 136, 'portugal': 137, 'qatar': 138, 'romania': 139, 'russia': 140, 'rwanda': 141, 'saint vincent & the grenadines': 142, 'samoa': 143, 'san marino': 144, 'sao tome & principe': 145, 'saudi arabia': 146, 'senegal': 147, 'serbia': 148, 'seychelles': 149, 'sierra leone': 150, 'singapore': 151, 'slovakia': 152, 'slovenia': 153, 'solomon islands': 154, 'somalia': 155, 'south africa': 156, 'south korea': 157, 'south sudan': 158, 'spain': 159, 'sri lanka': 160, 'st kitts & nevis': 161, 'st lucia': 162, 'sudan': 163, 'suriname': 164, 'swaziland': 165, 'sweden': 166, 'switzerland': 167, 'syria': 168, 'taiwan': 169, 'tajikistan': 170, 'tanzania': 171, 'thailand': 172, 'togo': 173, 'tonga': 174, 'trinidad & tobago': 175, 'tunisia': 176, 'turkey': 177, 'turkmenistan': 178, 'tuvalu': 179, 'uganda': 180, 'uk': 181, 'ukraine': 182, 'united arab emirates': 183, 'uruguay': 184, 'us': 185, 'uzbekistan': 186, 'vanuatu': 187, 'vatican city': 188, 'venezuela': 189, 'vietnam': 190, 'yemen': 191, 'zambia': 192, 'zimbabwe': 193},

    'state' : {'alabama': 194, 'alaska': 195, 'alberta': 196, 'arizona': 197, 'arkansas': 198, 'british columbia': 199, 'california': 200, 'colorado': 201, 'connecticut': 202, 'delaware': 203, 'district of columbia': 204, 'florida': 205, 'georgia': 206, 'hawaii': 207, 'idaho': 208, 'illinois': 209, 'indiana': 210, 'iowa': 211, 'kansas': 212, 'kentucky': 213, 'louisiana': 214, 'maine': 215, 'manitoba': 216, 'maryland': 217, 'massachusetts': 218, 'michigan': 219, 'minnesota': 220, 'mississippi': 221, 'missouri': 222, 'montana': 223, 'nebraska': 224, 'nevada': 225, 'new brunswick': 226, 'new hampshire': 227, 'new jersey': 228, 'new mexico': 229, 'new york': 230, 'newfoundland and labrador': 231, 'north carolina': 232, 'north dakota': 233, 'northwest territories': 234, 'nova scotia': 235, 'nunavut': 236, 'ohio': 237, 'oklahoma': 238, 'ontario': 239, 'oregon': 240, 'pennsylvania': 241, 'prince edward island': 242, 'quebec': 243, 'rhode island': 244, 'saskatchewan': 245, 'south carolina': 246, 'south dakota': 247, 'tennessee': 248, 'texas': 249, 'utah': 250, 'vermont': 251, 'virginia': 252, 'washington': 253, 'west virginia': 254, 'wisconsin': 255, 'wyoming': 256, 'yukon': 257},

    'gender' : {'male' : 258, 'female' : 259},

    'age' : {'u18' : 260, 'u25' : 261, 'o25': 262},
}

contextsize = sum([len(contextsid[key]) for key in contextsid])

def printContexts(contexts):
    result = '{\n'
    
    for type in contexts:
        result += "'{}' : ".format(type)
        result += "{"
        for context in contexts[type]:
            result += "'{}': {}, ".format(context, contexts[type][context])
        result += "},\n\n "

    result += "\n}"

    return result

def printContextsWithNum(contexts, oldcontexts):
    result = '{\n'

    for type in contexts:
        for context in contexts[type]:
            result += "{}: {}, ".format(oldcontexts[type][context], contexts[type][context])

    result += "\n}"

    return result
