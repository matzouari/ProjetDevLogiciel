export default `:host(.scrollable){overflow:auto;}:host(.scrollable-vertical){overflow-y:auto;}:host(.scrollable-horizontal){overflow-x:auto;}.scroll-button{position:sticky;top:calc(100% - 38px);left:calc(100% - 60px);cursor:pointer;visibility:hidden;font-size:18px;border-radius:50%;background-color:rgba(0, 0, 0, 0.25);color:white;width:36px;min-height:36px;margin-bottom:-36px;display:flex;align-items:center;justify-content:center;z-index:9999;opacity:0;transition:visibility 0s,
    opacity 0.2s ease-in-out;}.visible{visibility:visible;opacity:1;}.scroll-button:before{content:'⬇';}`
