if(EXISTS "/Users/rtyf/Documents/Documents - Miyoko/School/kth/wasp/tda ht23/masspcf.nosync/mpcf_test[1]_tests.cmake")
  include("/Users/rtyf/Documents/Documents - Miyoko/School/kth/wasp/tda ht23/masspcf.nosync/mpcf_test[1]_tests.cmake")
else()
  add_test(mpcf_test_NOT_BUILT mpcf_test_NOT_BUILT)
endif()
