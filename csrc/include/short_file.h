#include <string>

#ifdef __unix__
#define PATH_DELIMITER '/'
#else
#ifdef DINOML_HIP
#define PATH_DELIMITER '/'
#else
#define PATH_DELIMITER '\\'
#endif
#endif

#define __SHORT_FILE__ (strrchr(__FILE__, PATH_DELIMITER) ? \
                        strrchr(__FILE__, PATH_DELIMITER) + 1 : \
                        __FILE__)
