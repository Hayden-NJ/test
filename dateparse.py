import dateutil
def tryparse(date):
    kwargs = {}
    if isinstance(date, (tuple, list)):
        date = ''.join([str(x) for x in date])
    elif isinstance(date, int):
        date = str(date)
    elif isinstance(date, dict):
        kwargs = date
        date = kwargs.pop('date')
    try:
        try:
            parsedate = dateutil.parser.parse(date, **kwargs)
#             print('Sharp %r -> %s' % (date, parsedate))
        except ValueError:
            parsedate = dateutil.parser.parse(date, fuzzy=True, **kwargs)
#             print('Fuzzy %r -> %s' % (date, parsedate))
        return parsedate
    except Exception as err:
#         print("Try as I may, I cann't parse %r (%s)" % (date, err))
        return 'failed'
