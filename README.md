# drainrs
drainrs implements the [Drain](https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf) algorithm for automatic log parsing.

# Example:
```
 cargo run ./apache-short.log | tail
{"template":"[Sat Jun <*> <*> <*> [error] [client <*> script not found or unable to stat: /var/www/cgi-bin/awstats",
"values":["11","03:03:04","2005]","202.133.98.6]"]}
{"template":"[Sat Jun <*> <*> <*> [error] [client <*> script not found or unable to stat: /var/www/cgi-bin/awstats",
"values":["11","03:03:04","2005]","202.133.98.6]"]}
{"template":"[Sat Jun <*> <*> <*> [error] [client <*> script not found or unable to stat: /var/www/cgi-bin/awstats",
"values":["11","03:03:04","2005]","202.133.98.6]"]}
{"template":"[Sat Jun <*> <*> <*> [error] <*> Can't find child <*> in scoreboard",
"values":["11","03:03:04","2005]","jk2_init()","4210"]}
{"template":"[Sat Jun <*> <*> <*> [notice] workerEnv.init() ok <*>",
"values":["11","03:03:04","2005]","/etc/httpd/conf/workers2.properties"]}
{"template":"[Sat Jun <*> <*> <*> [error] mod_jk child init <*> <*>",
"values":["11","03:03:04","2005]","1","-2"]}
{"template":"[Sat Jun <*> <*> <*> [error] [client <*> script not found or unable to stat: /var/www/cgi-bin/awstats",
"values":["11","03:03:04","2005]","202.133.98.6]"]}
```

# Vocabulary
A **log record** is an entry in a text file, typically one-line but doesn't have to be.

  E.g. `[Thu Jun 09 06:07:05 2005] [notice] Digest logline here: done`


A **log template** is the string template that was used to format log record.

For example in Python format, that would look like this:

  `"[{date}] [{log_level}] Digest logline here: {status}".format(...)`

Or in the syntax output by drain.py and drain3.py:

  `"[<date>] [<log_level>] Digest logline here: <*>"`

# TODO
* None of the parameters that are configurable in the Python version are yet configurable here.
* The first drain allowed `split_line_provided`, which let you write a simple token-mapper like this:

  `<timestamp> <loglevel> <content>`

And then drain would only apply its logic to `<content>`.

Drain3 appears to have dropped this in favor of preprocessing on the user-code side, which is fair enough, although
the feature is very helpful from a cli/no-coding perspective.

* Drain3 allows "masking", which appears to be for recognizing values like IPs or numbers.
We have preliminary support for masking but it's not configurable from outside of the class and the user interface
to it is not yet defined.
