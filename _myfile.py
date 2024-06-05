<!DOCTYPE HTML>
<html>

<head>
    <meta charset="utf-8">

    <title>myfile.py (editing)</title>
    <link id="favicon" rel="shortcut icon" type="image/x-icon" href="/static/base/images/favicon-file.ico?v=f9f0a782d7d67b3a57bf7dce251d771b405c7890604576ec8b9a621a39d7670f6b43ffabef1e566f1cd741ee302e15977d9e1cf60bbacebafe75787b9916415c">
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <link rel="stylesheet" href="/static/components/jquery-ui/dist/themes/smoothness/jquery-ui.min.css?v=32f9dcde0cd9843f2b66d34c1c9928b59a5d7ef007ba7a6a6a790b3e78f7857a698444d7a716dfaf8fa834c3b3175efd258bbc07cfc4aabb86769b07e5f358c3" type="text/css" />
    <link rel="stylesheet" href="/static/components/jquery-typeahead/dist/jquery.typeahead.min.css?v=5edf53bf6bb9c3b1ddafd8594825a7e2ed621f19423e569c985162742f63911c09eba2c529f8fb47aebf27fafdfe287d563347f58c1126b278189a18871b6a9a" type="text/css" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    
<link rel="stylesheet" href="/static/components/codemirror/lib/codemirror.css?v=a545ad5e21a51420a7cb40234688eef087a5cf3798f64d7750291a8be0e9c760b8a1c9cbbbdcaa6470f2f385caa59e816f2640f609d29147f4762e27f69709e6">
<link rel="stylesheet" href="/static/components/codemirror/addon/dialog/dialog.css?v=e7f5db4afaccf8a00c10d62c3693642323d3fcf17604a4797803df04e4f144a944dc646c2fda20548df538ada775570127b2a15403996a171ced1769697f3ee4">

    <link rel="stylesheet" href="/static/style/style.min.css?v=e1ab1c38b672063a6541baf468c83345cd0f509729783ec9b7ccb64073004f5f056110c82c28aefbf3dbf32e0e040f05b8f0420bc411b669ed3d4f07511812ca" type="text/css"/>
    

    <link rel="stylesheet" href="/custom/custom.css" type="text/css" />
    <script src="/static/components/es6-promise/promise.min.js?v=bea335d74136a63ae1b5130f5ac9a50c6256a5f435e6e09fef599491a84d834a8b0f011ca3eaaca3b4ab6a2da2d3e1191567a2f171e60da1d10e5b9d52f84184" type="text/javascript" charset="utf-8"></script>
    <script src="/static/components/react/react.production.min.js?v=9a0aaf84a316c8bedd6c2ff7d5b5e0a13f8f84ec02442346cba0b842c6c81a6bf6176e64f3675c2ebf357cb5bb048e0b527bd39377c95681d22468da3d5de735" type="text/javascript"></script>
    <script src="/static/components/react/react-dom.production.min.js?v=6fc58c1c4736868ff84f57bd8b85f2bdb985993a9392718f3b4af4bfa10fb4efba2b4ddd68644bd2a8daf0619a3844944c9c43f8528364a1aa6fc01ec1b8ae84" type="text/javascript"></script>
    <script src="/static/components/create-react-class/index.js?v=894ad57246e682b4cfbe7cd5e408dcd6b38d06af4de4f3425991e2676fdc2ef1732cbd19903104198878ae77de12a1996de3e7da3a467fb226bdda8f4618faec" type="text/javascript"></script>
    <script src="/static/components/requirejs/require.js?v=d37b48bb2137faa0ab98157e240c084dd5b1b5e74911723aa1d1f04c928c2a03dedf922d049e4815f7e5a369faa2e6b6a1000aae958b7953b5cc60411154f593" type="text/javascript" charset="utf-8"></script>
    <script>
      require.config({
          
          urlArgs: "v=20240603151817",
          
          baseUrl: '/static/',
          paths: {
            'auth/js/main': 'auth/js/main.min',
            custom : '/custom',
            nbextensions : '/nbextensions',
            kernelspecs : '/kernelspecs',
            underscore : 'components/underscore/underscore-min',
            backbone : 'components/backbone/backbone-min',
            jed: 'components/jed/jed',
            jquery: 'components/jquery/jquery.min',
            json: 'components/requirejs-plugins/src/json',
            text: 'components/requirejs-text/text',
            bootstrap: 'components/bootstrap/dist/js/bootstrap.min',
            bootstraptour: 'components/bootstrap-tour/build/js/bootstrap-tour.min',
            'jquery-ui': 'components/jquery-ui/dist/jquery-ui.min',
            moment: 'components/moment/min/moment-with-locales',
            codemirror: 'components/codemirror',
            termjs: 'components/xterm.js/xterm',
            typeahead: 'components/jquery-typeahead/dist/jquery.typeahead.min',
          },
          map: { // for backward compatibility
              "*": {
                  "jqueryui": "jquery-ui",
              }
          },
          shim: {
            typeahead: {
              deps: ["jquery"],
              exports: "typeahead"
            },
            underscore: {
              exports: '_'
            },
            backbone: {
              deps: ["underscore", "jquery"],
              exports: "Backbone"
            },
            bootstrap: {
              deps: ["jquery"],
              exports: "bootstrap"
            },
            bootstraptour: {
              deps: ["bootstrap"],
              exports: "Tour"
            },
            "jquery-ui": {
              deps: ["jquery"],
              exports: "$"
            }
          },
          waitSeconds: 30,
      });

      require.config({
          map: {
              '*':{
                'contents': 'services/contents',
              }
          }
      });

      // error-catching custom.js shim.
      define("custom", function (require, exports, module) {
          try {
              var custom = require('custom/custom');
              console.debug('loaded custom.js');
              return custom;
          } catch (e) {
              console.error("error loading custom.js", e);
              return {};
          }
      })

      // error-catching custom-preload.js shim.
      define("custom-preload", function (require, exports, module) {
          try {
              var custom = require('custom/custom-preload');
              console.debug('loaded custom-preload.js');
              return custom;
          } catch (e) {
              console.error("error loading custom-preload.js", e);
              return {};
          }
      })

    document.nbjs_translations = {"domain": "nbjs", "locale_data": {"nbjs": {"": {"domain": "nbjs"}}}};
    document.documentElement.lang = navigator.language.toLowerCase();
    </script>

    
    

</head>

<body class="edit_app "
 
data-base-url="/"
data-file-path="MINI/myfile.py"

  
    data-jupyter-api-token="9212d1c8521cc3cfc4ee2663b708c92567ab035cd4d13a1b"
  
 

dir="ltr">

<noscript>
    <div id='noscript'>
      Jupyter Notebook requires JavaScript.<br>
      Please enable it to proceed. 
  </div>
</noscript>

<div id="header" role="navigation" aria-label="Top Menu">
  <div  id="newsId" style="display: none">
    
    <div class="alert alert-info" role="alert">
      <div style="display: flex">
        <div>
          <span class="label label-warning">UPDATE</span>
          Read <a href="https://jupyter-notebook.readthedocs.io/en/latest/migrate_to_notebook7.html" style="text-decoration: underline;" target="_blank">the migration plan</a> to Notebook 7 to learn about the new features and the actions to take if you are using extensions
          -
          Please note that updating to Notebook 7 might break some of your extensions.
        </div>
        <div style="margin-left: auto;">
          <a href="" onclick="alert('This message will not be shown anymore.'); return false;">
            <button type="button" class="btn btn-default btn-xs" id="dontShowId">
              Don't show anymore
            </button>
          </a>
        </div>
      </div>
    </div>
    
  </div>
  <div id="header-container" class="container">
  <div id="ipython_notebook" class="nav navbar-brand"><a href="/tree?token=9212d1c8521cc3cfc4ee2663b708c92567ab035cd4d13a1b" title='dashboard'>
      <img src='/static/base/images/logo.png?v=a2a176ee3cee251ffddf5fa21fe8e43727a9e5f87a06f9c91ad7b776d9e9d3d5e0159c16cc188a3965e00375fb4bc336c16067c688f5040c0c2d4bfdb852a9e4' alt='Jupyter Notebook'/>
  </a></div>

  

<span id="save_widget" class="pull-left save_widget">
    <span class="filename"></span>
    <span class="last_modified"></span>
</span>


  

  
  
  
  

    <span id="login_widget">
      
        <button id="logout" class="btn btn-sm navbar-btn">Logout</button>
      
    </span>

  

  
  
  </div>
  <div class="header-bar"></div>

  

<div id="menubar-container" class="container">
  <div id="menubar">
    <div id="menus" class="navbar navbar-default" role="navigation">
      <div class="container-fluid">
          <p  class="navbar-text indicator_area">
          <span id="current-mode" >current mode</span>
          </p>
        <button type="button" class="btn btn-default navbar-toggle" data-toggle="collapse" data-target=".navbar-collapse">
          <i class="fa fa-bars"></i>
          <span class="navbar-text">Menu</span>
        </button>
        <ul class="nav navbar-nav navbar-right">
          <li id="notification_area"></li>
        </ul>
        <div class="navbar-collapse collapse">
          <ul class="nav navbar-nav">
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">File</a>
              <ul id="file-menu" class="dropdown-menu">
                <li id="new-file"><a href="#">New</a></li>
                <li id="save-file"><a href="#">Save</a></li>
                <li id="rename-file"><a href="#">Rename</a></li>
                <li id="download-file"><a href="#">Download</a></li>
              </ul>
            </li>
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">Edit</a>
              <ul id="edit-menu" class="dropdown-menu">
                <li id="menu-find"><a href="#">Find</a></li>
                <li id="menu-replace"><a href="#">Find &amp; Replace</a></li>
                <li class="divider"></li>
                <li class="dropdown-header">Key Map</li>
                <li id="menu-keymap-default"><a href="#">Default<i class="fa"></i></a></li>
                <li id="menu-keymap-sublime"><a href="#">Sublime Text<i class="fa"></i></a></li>
                <li id="menu-keymap-vim"><a href="#">Vim<i class="fa"></i></a></li>
                <li id="menu-keymap-emacs"><a href="#">emacs<i class="fa"></i></a></li>
              </ul>
            </li>
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">View</a>
              <ul id="view-menu" class="dropdown-menu">
              <li id="toggle_header" title="Show/Hide the logo and notebook title (above menu bar)">
              <a href="#">Toggle Header</a></li>
              <li id="menu-line-numbers"><a href="#">Toggle Line Numbers</a></li>
              </ul>
            </li>
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">Language</a>
              <ul id="mode-menu" class="dropdown-menu">
              </ul>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="lower-header-bar"></div>


</div>

<div id="site">


<div id="texteditor-backdrop">
<div id="texteditor-container" class="container"></div>
</div>


</div>






    


<script src="/static/edit/js/main.min.js?v=e0042c0d539a8219218d83395056db19eeba104af064b6abac688381af3f508ff7086aaa146696f73086a76271ff3171db5332e8a3ab863fd52a697de73dfc0d" type="text/javascript" charset="utf-8"></script>


<script type='text/javascript'>
  function _remove_token_from_url() {
    if (window.location.search.length <= 1) {
      return;
    }
    var search_parameters = window.location.search.slice(1).split('&');
    for (var i = 0; i < search_parameters.length; i++) {
      if (search_parameters[i].split('=')[0] === 'token') {
        // remote token from search parameters
        search_parameters.splice(i, 1);
        var new_search = '';
        if (search_parameters.length) {
          new_search = '?' + search_parameters.join('&');
        }
        var new_url = window.location.origin + 
                      window.location.pathname + 
                      new_search + 
                      window.location.hash;
        window.history.replaceState({}, "", new_url);
        return;
      }
    }
  }
  _remove_token_from_url();
  sys_info = {"notebook_version": "6.5.4", "notebook_path": "C:\\Users\\91938\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\notebook", "commit_source": "", "commit_hash": "", "sys_version": "3.11.3 (tags/v3.11.3:f3909b8, Apr  4 2023, 23:49:59) [MSC v.1934 64 bit (AMD64)]", "sys_executable": "C:\\Users\\91938\\AppData\\Local\\Programs\\Python\\Python311\\python.exe", "sys_platform": "win32", "platform": "Windows-10-10.0.22631-SP0", "os_name": "nt", "default_encoding": "utf-8"};
  document.addEventListener('DOMContentLoaded', function () {
    const newsId = document.querySelector('#newsId');
    const dontShowId = document.querySelector('#dontShowId');
    const showNotebookNews = localStorage.getItem('showNotebookNews');
    dontShowId.addEventListener('click', () => {
      localStorage.setItem('showNotebookNews', false);
      newsId.style.display = 'none';
    });
    if (!showNotebookNews) newsId.style.display = 'inline';
  });
</script>
</body>

</html>