__rethrow_casa_exceptions = True
h_init()
try:
  hifv_restoredata (vis=['../rawdata/mySDM'], session=['session_1'],\
                   ocorr_mode='co',gainmap=False)
  hifv_statwt()
finally:
  h_save()
