# Created in 2025 by Gandecheng
# Used to store POSSIBLE intent, must in a format as follows:
# 'Wireless Newtwork type':{
#     'option1 in access process':{
#         'fixed': None,
#         'varies': ['begin', 'end'],
#         'other options': 'corresponding range'
#     }
#     'option2 in access process':{
#         ...
#     }
# }

INTENT_CONFIG={
    'NB-IoT':{
        'access probability': {
            'fixed': None,
            'varies': [0,1],
        }
    },
}