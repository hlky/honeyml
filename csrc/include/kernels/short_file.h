#ifdef __unix__
#define PATH_DELIMITER '/'
#else
#define PATH_DELIMITER '\\'
#endif

#define __SHORT_FILE__ (strrchr(__FILE__, PATH_DELIMITER) ? \
                        strrchr(__FILE__, PATH_DELIMITER) + 1 : \
                        __FILE__)
