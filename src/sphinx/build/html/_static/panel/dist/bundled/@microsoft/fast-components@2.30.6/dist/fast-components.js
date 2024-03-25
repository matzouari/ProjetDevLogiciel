/**
 * A reference to globalThis, with support
 * for browsers that don't yet support the spec.
 * @public
 */
const $global = function () {
  if (typeof globalThis !== "undefined") {
    // We're running in a modern environment.
    return globalThis;
  }

  if (typeof global !== "undefined") {
    // We're running in NodeJS
    return global;
  }

  if (typeof self !== "undefined") {
    // We're running in a worker.
    return self;
  }

  if (typeof window !== "undefined") {
    // We're running in the browser's main thread.
    return window;
  }

  try {
    // Hopefully we never get here...
    // Not all environments allow eval and Function. Use only as a last resort:
    // eslint-disable-next-line no-new-func
    return new Function("return this")();
  } catch (_a) {
    // If all fails, give up and create an object.
    // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
    return {};
  }
}(); // API-only Polyfill for trustedTypes

if ($global.trustedTypes === void 0) {
  $global.trustedTypes = {
    createPolicy: (n, r) => r
  };
}

const propConfig = {
  configurable: false,
  enumerable: false,
  writable: false
};

if ($global.FAST === void 0) {
  Reflect.defineProperty($global, "FAST", Object.assign({
    value: Object.create(null)
  }, propConfig));
}
/**
 * The FAST global.
 * @internal
 */


const FAST = $global.FAST;

if (FAST.getById === void 0) {
  const storage = Object.create(null);
  Reflect.defineProperty(FAST, "getById", Object.assign({
    value(id, initialize) {
      let found = storage[id];

      if (found === void 0) {
        found = initialize ? storage[id] = initialize() : null;
      }

      return found;
    }

  }, propConfig));
}
/**
 * A readonly, empty array.
 * @remarks
 * Typically returned by APIs that return arrays when there are
 * no actual items to return.
 * @internal
 */


const emptyArray = Object.freeze([]);

const updateQueue = $global.FAST.getById(1
/* updateQueue */
, () => {
  const tasks = [];
  const pendingErrors = [];

  function throwFirstError() {
    if (pendingErrors.length) {
      throw pendingErrors.shift();
    }
  }

  function tryRunTask(task) {
    try {
      task.call();
    } catch (error) {
      pendingErrors.push(error);
      setTimeout(throwFirstError, 0);
    }
  }

  function process() {
    const capacity = 1024;
    let index = 0;

    while (index < tasks.length) {
      tryRunTask(tasks[index]);
      index++; // Prevent leaking memory for long chains of recursive calls to `DOM.queueUpdate`.
      // If we call `DOM.queueUpdate` within a task scheduled by `DOM.queueUpdate`, the queue will
      // grow, but to avoid an O(n) walk for every task we execute, we don't
      // shift tasks off the queue after they have been executed.
      // Instead, we periodically shift 1024 tasks off the queue.

      if (index > capacity) {
        // Manually shift all values starting at the index back to the
        // beginning of the queue.
        for (let scan = 0, newLength = tasks.length - index; scan < newLength; scan++) {
          tasks[scan] = tasks[scan + index];
        }

        tasks.length -= index;
        index = 0;
      }
    }

    tasks.length = 0;
  }

  function enqueue(callable) {
    if (tasks.length < 1) {
      $global.requestAnimationFrame(process);
    }

    tasks.push(callable);
  }

  return Object.freeze({
    enqueue,
    process
  });
});
/* eslint-disable */

const fastHTMLPolicy = $global.trustedTypes.createPolicy("fast-html", {
  createHTML: html => html
});
/* eslint-enable */

let htmlPolicy = fastHTMLPolicy;
const marker = `fast-${Math.random().toString(36).substring(2, 8)}`;
/** @internal */

const _interpolationStart = `${marker}{`;
/** @internal */

const _interpolationEnd = `}${marker}`;
/**
 * Common DOM APIs.
 * @public
 */

const DOM = Object.freeze({
  /**
   * Indicates whether the DOM supports the adoptedStyleSheets feature.
   */
  supportsAdoptedStyleSheets: Array.isArray(document.adoptedStyleSheets) && "replace" in CSSStyleSheet.prototype,

  /**
   * Sets the HTML trusted types policy used by the templating engine.
   * @param policy - The policy to set for HTML.
   * @remarks
   * This API can only be called once, for security reasons. It should be
   * called by the application developer at the start of their program.
   */
  setHTMLPolicy(policy) {
    if (htmlPolicy !== fastHTMLPolicy) {
      throw new Error("The HTML policy can only be set once.");
    }

    htmlPolicy = policy;
  },

  /**
   * Turns a string into trusted HTML using the configured trusted types policy.
   * @param html - The string to turn into trusted HTML.
   * @remarks
   * Used internally by the template engine when creating templates
   * and setting innerHTML.
   */
  createHTML(html) {
    return htmlPolicy.createHTML(html);
  },

  /**
   * Determines if the provided node is a template marker used by the runtime.
   * @param node - The node to test.
   */
  isMarker(node) {
    return node && node.nodeType === 8 && node.data.startsWith(marker);
  },

  /**
   * Given a marker node, extract the {@link HTMLDirective} index from the placeholder.
   * @param node - The marker node to extract the index from.
   */
  extractDirectiveIndexFromMarker(node) {
    return parseInt(node.data.replace(`${marker}:`, ""));
  },

  /**
   * Creates a placeholder string suitable for marking out a location *within*
   * an attribute value or HTML content.
   * @param index - The directive index to create the placeholder for.
   * @remarks
   * Used internally by binding directives.
   */
  createInterpolationPlaceholder(index) {
    return `${_interpolationStart}${index}${_interpolationEnd}`;
  },

  /**
   * Creates a placeholder that manifests itself as an attribute on an
   * element.
   * @param attributeName - The name of the custom attribute.
   * @param index - The directive index to create the placeholder for.
   * @remarks
   * Used internally by attribute directives such as `ref`, `slotted`, and `children`.
   */
  createCustomAttributePlaceholder(attributeName, index) {
    return `${attributeName}="${this.createInterpolationPlaceholder(index)}"`;
  },

  /**
   * Creates a placeholder that manifests itself as a marker within the DOM structure.
   * @param index - The directive index to create the placeholder for.
   * @remarks
   * Used internally by structural directives such as `repeat`.
   */
  createBlockPlaceholder(index) {
    return `<!--${marker}:${index}-->`;
  },

  /**
   * Schedules DOM update work in the next async batch.
   * @param callable - The callable function or object to queue.
   */
  queueUpdate: updateQueue.enqueue,

  /**
   * Immediately processes all work previously scheduled
   * through queueUpdate.
   * @remarks
   * This also forces nextUpdate promises
   * to resolve.
   */
  processUpdates: updateQueue.process,

  /**
   * Resolves with the next DOM update.
   */
  nextUpdate() {
    return new Promise(updateQueue.enqueue);
  },

  /**
   * Sets an attribute value on an element.
   * @param element - The element to set the attribute value on.
   * @param attributeName - The attribute name to set.
   * @param value - The value of the attribute to set.
   * @remarks
   * If the value is `null` or `undefined`, the attribute is removed, otherwise
   * it is set to the provided value using the standard `setAttribute` API.
   */
  setAttribute(element, attributeName, value) {
    if (value === null || value === undefined) {
      element.removeAttribute(attributeName);
    } else {
      element.setAttribute(attributeName, value);
    }
  },

  /**
   * Sets a boolean attribute value.
   * @param element - The element to set the boolean attribute value on.
   * @param attributeName - The attribute name to set.
   * @param value - The value of the attribute to set.
   * @remarks
   * If the value is true, the attribute is added; otherwise it is removed.
   */
  setBooleanAttribute(element, attributeName, value) {
    value ? element.setAttribute(attributeName, "") : element.removeAttribute(attributeName);
  },

  /**
   * Removes all the child nodes of the provided parent node.
   * @param parent - The node to remove the children from.
   */
  removeChildNodes(parent) {
    for (let child = parent.firstChild; child !== null; child = parent.firstChild) {
      parent.removeChild(child);
    }
  },

  /**
   * Creates a TreeWalker configured to walk a template fragment.
   * @param fragment - The fragment to walk.
   */
  createTemplateWalker(fragment) {
    return document.createTreeWalker(fragment, 133, // element, text, comment
    null, false);
  }

});

function spilloverSubscribe(subscriber) {
  const spillover = this.spillover;
  const index = spillover.indexOf(subscriber);

  if (index === -1) {
    spillover.push(subscriber);
  }
}

function spilloverUnsubscribe(subscriber) {
  const spillover = this.spillover;
  const index = spillover.indexOf(subscriber);

  if (index !== -1) {
    spillover.splice(index, 1);
  }
}

function spilloverNotifySubscribers(args) {
  const spillover = this.spillover;
  const source = this.source;

  for (let i = 0, ii = spillover.length; i < ii; ++i) {
    spillover[i].handleChange(source, args);
  }
}

function spilloverHas(subscriber) {
  return this.spillover.indexOf(subscriber) !== -1;
}
/**
 * An implementation of {@link Notifier} that efficiently keeps track of
 * subscribers interested in a specific change notification on an
 * observable source.
 *
 * @remarks
 * This set is optimized for the most common scenario of 1 or 2 subscribers.
 * With this in mind, it can store a subscriber in an internal field, allowing it to avoid Array#push operations.
 * If the set ever exceeds two subscribers, it upgrades to an array automatically.
 * @public
 */


class SubscriberSet {
  /**
   * Creates an instance of SubscriberSet for the specified source.
   * @param source - The object source that subscribers will receive notifications from.
   * @param initialSubscriber - An initial subscriber to changes.
   */
  constructor(source, initialSubscriber) {
    this.sub1 = void 0;
    this.sub2 = void 0;
    this.spillover = void 0;
    this.source = source;
    this.sub1 = initialSubscriber;
  }
  /**
   * Checks whether the provided subscriber has been added to this set.
   * @param subscriber - The subscriber to test for inclusion in this set.
   */


  has(subscriber) {
    return this.sub1 === subscriber || this.sub2 === subscriber;
  }
  /**
   * Subscribes to notification of changes in an object's state.
   * @param subscriber - The object that is subscribing for change notification.
   */


  subscribe(subscriber) {
    if (this.has(subscriber)) {
      return;
    }

    if (this.sub1 === void 0) {
      this.sub1 = subscriber;
      return;
    }

    if (this.sub2 === void 0) {
      this.sub2 = subscriber;
      return;
    }

    this.spillover = [this.sub1, this.sub2, subscriber];
    this.subscribe = spilloverSubscribe;
    this.unsubscribe = spilloverUnsubscribe;
    this.notify = spilloverNotifySubscribers;
    this.has = spilloverHas;
    this.sub1 = void 0;
    this.sub2 = void 0;
  }
  /**
   * Unsubscribes from notification of changes in an object's state.
   * @param subscriber - The object that is unsubscribing from change notification.
   */


  unsubscribe(subscriber) {
    if (this.sub1 === subscriber) {
      this.sub1 = void 0;
    } else if (this.sub2 === subscriber) {
      this.sub2 = void 0;
    }
  }
  /**
   * Notifies all subscribers.
   * @param args - Data passed along to subscribers during notification.
   */


  notify(args) {
    const sub1 = this.sub1;
    const sub2 = this.sub2;
    const source = this.source;

    if (sub1 !== void 0) {
      sub1.handleChange(source, args);
    }

    if (sub2 !== void 0) {
      sub2.handleChange(source, args);
    }
  }

}
/**
 * An implementation of Notifier that allows subscribers to be notified
 * of individual property changes on an object.
 * @public
 */

class PropertyChangeNotifier {
  /**
   * Creates an instance of PropertyChangeNotifier for the specified source.
   * @param source - The object source that subscribers will receive notifications from.
   */
  constructor(source) {
    this.subscribers = {};
    this.sourceSubscribers = null;
    this.source = source;
  }
  /**
   * Notifies all subscribers, based on the specified property.
   * @param propertyName - The property name, passed along to subscribers during notification.
   */


  notify(propertyName) {
    var _a;

    const subscribers = this.subscribers[propertyName];

    if (subscribers !== void 0) {
      subscribers.notify(propertyName);
    }

    (_a = this.sourceSubscribers) === null || _a === void 0 ? void 0 : _a.notify(propertyName);
  }
  /**
   * Subscribes to notification of changes in an object's state.
   * @param subscriber - The object that is subscribing for change notification.
   * @param propertyToWatch - The name of the property that the subscriber is interested in watching for changes.
   */


  subscribe(subscriber, propertyToWatch) {
    var _a;

    if (propertyToWatch) {
      let subscribers = this.subscribers[propertyToWatch];

      if (subscribers === void 0) {
        this.subscribers[propertyToWatch] = subscribers = new SubscriberSet(this.source);
      }

      subscribers.subscribe(subscriber);
    } else {
      this.sourceSubscribers = (_a = this.sourceSubscribers) !== null && _a !== void 0 ? _a : new SubscriberSet(this.source);
      this.sourceSubscribers.subscribe(subscriber);
    }
  }
  /**
   * Unsubscribes from notification of changes in an object's state.
   * @param subscriber - The object that is unsubscribing from change notification.
   * @param propertyToUnwatch - The name of the property that the subscriber is no longer interested in watching.
   */


  unsubscribe(subscriber, propertyToUnwatch) {
    var _a;

    if (propertyToUnwatch) {
      const subscribers = this.subscribers[propertyToUnwatch];

      if (subscribers !== void 0) {
        subscribers.unsubscribe(subscriber);
      }
    } else {
      (_a = this.sourceSubscribers) === null || _a === void 0 ? void 0 : _a.unsubscribe(subscriber);
    }
  }

}

/**
 * Common Observable APIs.
 * @public
 */

const Observable = FAST.getById(2
/* observable */
, () => {
  const volatileRegex = /(:|&&|\|\||if)/;
  const notifierLookup = new WeakMap();
  const accessorLookup = new WeakMap();
  const queueUpdate = DOM.queueUpdate;
  let watcher = void 0;

  let createArrayObserver = array => {
    throw new Error("Must call enableArrayObservation before observing arrays.");
  };

  function getNotifier(source) {
    let found = source.$fastController || notifierLookup.get(source);

    if (found === void 0) {
      if (Array.isArray(source)) {
        found = createArrayObserver(source);
      } else {
        notifierLookup.set(source, found = new PropertyChangeNotifier(source));
      }
    }

    return found;
  }

  function getAccessors(target) {
    let accessors = accessorLookup.get(target);

    if (accessors === void 0) {
      let currentTarget = Reflect.getPrototypeOf(target);

      while (accessors === void 0 && currentTarget !== null) {
        accessors = accessorLookup.get(currentTarget);
        currentTarget = Reflect.getPrototypeOf(currentTarget);
      }

      if (accessors === void 0) {
        accessors = [];
      } else {
        accessors = accessors.slice(0);
      }

      accessorLookup.set(target, accessors);
    }

    return accessors;
  }

  class DefaultObservableAccessor {
    constructor(name) {
      this.name = name;
      this.field = `_${name}`;
      this.callback = `${name}Changed`;
    }

    getValue(source) {
      if (watcher !== void 0) {
        watcher.watch(source, this.name);
      }

      return source[this.field];
    }

    setValue(source, newValue) {
      const field = this.field;
      const oldValue = source[field];

      if (oldValue !== newValue) {
        source[field] = newValue;
        const callback = source[this.callback];

        if (typeof callback === "function") {
          callback.call(source, oldValue, newValue);
        }

        getNotifier(source).notify(this.name);
      }
    }

  }

  class BindingObserverImplementation extends SubscriberSet {
    constructor(binding, initialSubscriber, isVolatileBinding = false) {
      super(binding, initialSubscriber);
      this.binding = binding;
      this.isVolatileBinding = isVolatileBinding;
      this.needsRefresh = true;
      this.needsQueue = true;
      this.first = this;
      this.last = null;
      this.propertySource = void 0;
      this.propertyName = void 0;
      this.notifier = void 0;
      this.next = void 0;
    }

    observe(source, context) {
      if (this.needsRefresh && this.last !== null) {
        this.disconnect();
      }

      const previousWatcher = watcher;
      watcher = this.needsRefresh ? this : void 0;
      this.needsRefresh = this.isVolatileBinding;
      const result = this.binding(source, context);
      watcher = previousWatcher;
      return result;
    }

    disconnect() {
      if (this.last !== null) {
        let current = this.first;

        while (current !== void 0) {
          current.notifier.unsubscribe(this, current.propertyName);
          current = current.next;
        }

        this.last = null;
        this.needsRefresh = this.needsQueue = true;
      }
    }

    watch(propertySource, propertyName) {
      const prev = this.last;
      const notifier = getNotifier(propertySource);
      const current = prev === null ? this.first : {};
      current.propertySource = propertySource;
      current.propertyName = propertyName;
      current.notifier = notifier;
      notifier.subscribe(this, propertyName);

      if (prev !== null) {
        if (!this.needsRefresh) {
          // Declaring the variable prior to assignment below circumvents
          // a bug in Angular's optimization process causing infinite recursion
          // of this watch() method. Details https://github.com/microsoft/fast/issues/4969
          let prevValue;
          watcher = void 0;
          /* eslint-disable-next-line */

          prevValue = prev.propertySource[prev.propertyName];
          /* eslint-disable-next-line @typescript-eslint/no-this-alias */

          watcher = this;

          if (propertySource === prevValue) {
            this.needsRefresh = true;
          }
        }

        prev.next = current;
      }

      this.last = current;
    }

    handleChange() {
      if (this.needsQueue) {
        this.needsQueue = false;
        queueUpdate(this);
      }
    }

    call() {
      if (this.last !== null) {
        this.needsQueue = true;
        this.notify(this);
      }
    }

    records() {
      let next = this.first;
      return {
        next: () => {
          const current = next;

          if (current === undefined) {
            return {
              value: void 0,
              done: true
            };
          } else {
            next = next.next;
            return {
              value: current,
              done: false
            };
          }
        },
        [Symbol.iterator]: function () {
          return this;
        }
      };
    }

  }

  return Object.freeze({
    /**
     * @internal
     * @param factory - The factory used to create array observers.
     */
    setArrayObserverFactory(factory) {
      createArrayObserver = factory;
    },

    /**
     * Gets a notifier for an object or Array.
     * @param source - The object or Array to get the notifier for.
     */
    getNotifier,

    /**
     * Records a property change for a source object.
     * @param source - The object to record the change against.
     * @param propertyName - The property to track as changed.
     */
    track(source, propertyName) {
      if (watcher !== void 0) {
        watcher.watch(source, propertyName);
      }
    },

    /**
     * Notifies watchers that the currently executing property getter or function is volatile
     * with respect to its observable dependencies.
     */
    trackVolatile() {
      if (watcher !== void 0) {
        watcher.needsRefresh = true;
      }
    },

    /**
     * Notifies subscribers of a source object of changes.
     * @param source - the object to notify of changes.
     * @param args - The change args to pass to subscribers.
     */
    notify(source, args) {
      getNotifier(source).notify(args);
    },

    /**
     * Defines an observable property on an object or prototype.
     * @param target - The target object to define the observable on.
     * @param nameOrAccessor - The name of the property to define as observable;
     * or a custom accessor that specifies the property name and accessor implementation.
     */
    defineProperty(target, nameOrAccessor) {
      if (typeof nameOrAccessor === "string") {
        nameOrAccessor = new DefaultObservableAccessor(nameOrAccessor);
      }

      getAccessors(target).push(nameOrAccessor);
      Reflect.defineProperty(target, nameOrAccessor.name, {
        enumerable: true,
        get: function () {
          return nameOrAccessor.getValue(this);
        },
        set: function (newValue) {
          nameOrAccessor.setValue(this, newValue);
        }
      });
    },

    /**
     * Finds all the observable accessors defined on the target,
     * including its prototype chain.
     * @param target - The target object to search for accessor on.
     */
    getAccessors,

    /**
     * Creates a {@link BindingObserver} that can watch the
     * provided {@link Binding} for changes.
     * @param binding - The binding to observe.
     * @param initialSubscriber - An initial subscriber to changes in the binding value.
     * @param isVolatileBinding - Indicates whether the binding's dependency list must be re-evaluated on every value evaluation.
     */
    binding(binding, initialSubscriber, isVolatileBinding = this.isVolatileBinding(binding)) {
      return new BindingObserverImplementation(binding, initialSubscriber, isVolatileBinding);
    },

    /**
     * Determines whether a binding expression is volatile and needs to have its dependency list re-evaluated
     * on every evaluation of the value.
     * @param binding - The binding to inspect.
     */
    isVolatileBinding(binding) {
      return volatileRegex.test(binding.toString());
    }

  });
});
/**
 * Decorator: Defines an observable property on the target.
 * @param target - The target to define the observable on.
 * @param nameOrAccessor - The property name or accessor to define the observable as.
 * @public
 */

function observable(target, nameOrAccessor) {
  Observable.defineProperty(target, nameOrAccessor);
}
/**
 * Decorator: Marks a property getter as having volatile observable dependencies.
 * @param target - The target that the property is defined on.
 * @param name - The property name.
 * @param name - The existing descriptor.
 * @public
 */

function volatile(target, name, descriptor) {
  return Object.assign({}, descriptor, {
    get: function () {
      Observable.trackVolatile();
      return descriptor.get.apply(this);
    }
  });
}
const contextEvent = FAST.getById(3
/* contextEvent */
, () => {
  let current = null;
  return {
    get() {
      return current;
    },

    set(event) {
      current = event;
    }

  };
});
/**
 * Provides additional contextual information available to behaviors and expressions.
 * @public
 */

class ExecutionContext {
  constructor() {
    /**
     * The index of the current item within a repeat context.
     */
    this.index = 0;
    /**
     * The length of the current collection within a repeat context.
     */

    this.length = 0;
    /**
     * The parent data object within a repeat context.
     */

    this.parent = null;
    /**
     * The parent execution context when in nested context scenarios.
     */

    this.parentContext = null;
  }
  /**
   * The current event within an event handler.
   */


  get event() {
    return contextEvent.get();
  }
  /**
   * Indicates whether the current item within a repeat context
   * has an even index.
   */


  get isEven() {
    return this.index % 2 === 0;
  }
  /**
   * Indicates whether the current item within a repeat context
   * has an odd index.
   */


  get isOdd() {
    return this.index % 2 !== 0;
  }
  /**
   * Indicates whether the current item within a repeat context
   * is the first item in the collection.
   */


  get isFirst() {
    return this.index === 0;
  }
  /**
   * Indicates whether the current item within a repeat context
   * is somewhere in the middle of the collection.
   */


  get isInMiddle() {
    return !this.isFirst && !this.isLast;
  }
  /**
   * Indicates whether the current item within a repeat context
   * is the last item in the collection.
   */


  get isLast() {
    return this.index === this.length - 1;
  }
  /**
   * Sets the event for the current execution context.
   * @param event - The event to set.
   * @internal
   */


  static setEvent(event) {
    contextEvent.set(event);
  }

}
Observable.defineProperty(ExecutionContext.prototype, "index");
Observable.defineProperty(ExecutionContext.prototype, "length");
/**
 * The default execution context used in binding expressions.
 * @public
 */

const defaultExecutionContext = Object.seal(new ExecutionContext());

/**
 * Instructs the template engine to apply behavior to a node.
 * @public
 */

class HTMLDirective {
  constructor() {
    /**
     * The index of the DOM node to which the created behavior will apply.
     */
    this.targetIndex = 0;
  }

}
/**
 * A {@link HTMLDirective} that targets a named attribute or property on a node.
 * @public
 */

class TargetedHTMLDirective extends HTMLDirective {
  constructor() {
    super(...arguments);
    /**
     * Creates a placeholder string based on the directive's index within the template.
     * @param index - The index of the directive within the template.
     */

    this.createPlaceholder = DOM.createInterpolationPlaceholder;
  }

}
/**
 * A directive that attaches special behavior to an element via a custom attribute.
 * @public
 */

class AttachedBehaviorHTMLDirective extends HTMLDirective {
  /**
   *
   * @param name - The name of the behavior; used as a custom attribute on the element.
   * @param behavior - The behavior to instantiate and attach to the element.
   * @param options - Options to pass to the behavior during creation.
   */
  constructor(name, behavior, options) {
    super();
    this.name = name;
    this.behavior = behavior;
    this.options = options;
  }
  /**
   * Creates a placeholder string based on the directive's index within the template.
   * @param index - The index of the directive within the template.
   * @remarks
   * Creates a custom attribute placeholder.
   */


  createPlaceholder(index) {
    return DOM.createCustomAttributePlaceholder(this.name, index);
  }
  /**
   * Creates a behavior for the provided target node.
   * @param target - The node instance to create the behavior for.
   * @remarks
   * Creates an instance of the `behavior` type this directive was constructed with
   * and passes the target and options to that `behavior`'s constructor.
   */


  createBehavior(target) {
    return new this.behavior(target, this.options);
  }

}

function normalBind(source, context) {
  this.source = source;
  this.context = context;

  if (this.bindingObserver === null) {
    this.bindingObserver = Observable.binding(this.binding, this, this.isBindingVolatile);
  }

  this.updateTarget(this.bindingObserver.observe(source, context));
}

function triggerBind(source, context) {
  this.source = source;
  this.context = context;
  this.target.addEventListener(this.targetName, this);
}

function normalUnbind() {
  this.bindingObserver.disconnect();
  this.source = null;
  this.context = null;
}

function contentUnbind() {
  this.bindingObserver.disconnect();
  this.source = null;
  this.context = null;
  const view = this.target.$fastView;

  if (view !== void 0 && view.isComposed) {
    view.unbind();
    view.needsBindOnly = true;
  }
}

function triggerUnbind() {
  this.target.removeEventListener(this.targetName, this);
  this.source = null;
  this.context = null;
}

function updateAttributeTarget(value) {
  DOM.setAttribute(this.target, this.targetName, value);
}

function updateBooleanAttributeTarget(value) {
  DOM.setBooleanAttribute(this.target, this.targetName, value);
}

function updateContentTarget(value) {
  // If there's no actual value, then this equates to the
  // empty string for the purposes of content bindings.
  if (value === null || value === undefined) {
    value = "";
  } // If the value has a "create" method, then it's a template-like.


  if (value.create) {
    this.target.textContent = "";
    let view = this.target.$fastView; // If there's no previous view that we might be able to
    // reuse then create a new view from the template.

    if (view === void 0) {
      view = value.create();
    } else {
      // If there is a previous view, but it wasn't created
      // from the same template as the new value, then we
      // need to remove the old view if it's still in the DOM
      // and create a new view from the template.
      if (this.target.$fastTemplate !== value) {
        if (view.isComposed) {
          view.remove();
          view.unbind();
        }

        view = value.create();
      }
    } // It's possible that the value is the same as the previous template
    // and that there's actually no need to compose it.


    if (!view.isComposed) {
      view.isComposed = true;
      view.bind(this.source, this.context);
      view.insertBefore(this.target);
      this.target.$fastView = view;
      this.target.$fastTemplate = value;
    } else if (view.needsBindOnly) {
      view.needsBindOnly = false;
      view.bind(this.source, this.context);
    }
  } else {
    const view = this.target.$fastView; // If there is a view and it's currently composed into
    // the DOM, then we need to remove it.

    if (view !== void 0 && view.isComposed) {
      view.isComposed = false;
      view.remove();

      if (view.needsBindOnly) {
        view.needsBindOnly = false;
      } else {
        view.unbind();
      }
    }

    this.target.textContent = value;
  }
}

function updatePropertyTarget(value) {
  this.target[this.targetName] = value;
}

function updateClassTarget(value) {
  const classVersions = this.classVersions || Object.create(null);
  const target = this.target;
  let version = this.version || 0; // Add the classes, tracking the version at which they were added.

  if (value !== null && value !== undefined && value.length) {
    const names = value.split(/\s+/);

    for (let i = 0, ii = names.length; i < ii; ++i) {
      const currentName = names[i];

      if (currentName === "") {
        continue;
      }

      classVersions[currentName] = version;
      target.classList.add(currentName);
    }
  }

  this.classVersions = classVersions;
  this.version = version + 1; // If this is the first call to add classes, there's no need to remove old ones.

  if (version === 0) {
    return;
  } // Remove classes from the previous version.


  version -= 1;

  for (const name in classVersions) {
    if (classVersions[name] === version) {
      target.classList.remove(name);
    }
  }
}
/**
 * A directive that configures data binding to element content and attributes.
 * @public
 */


class HTMLBindingDirective extends TargetedHTMLDirective {
  /**
   * Creates an instance of BindingDirective.
   * @param binding - A binding that returns the data used to update the DOM.
   */
  constructor(binding) {
    super();
    this.binding = binding;
    this.bind = normalBind;
    this.unbind = normalUnbind;
    this.updateTarget = updateAttributeTarget;
    this.isBindingVolatile = Observable.isVolatileBinding(this.binding);
  }
  /**
   * Gets/sets the name of the attribute or property that this
   * binding is targeting.
   */


  get targetName() {
    return this.originalTargetName;
  }

  set targetName(value) {
    this.originalTargetName = value;

    if (value === void 0) {
      return;
    }

    switch (value[0]) {
      case ":":
        this.cleanedTargetName = value.substr(1);
        this.updateTarget = updatePropertyTarget;

        if (this.cleanedTargetName === "innerHTML") {
          const binding = this.binding;

          this.binding = (s, c) => DOM.createHTML(binding(s, c));
        }

        break;

      case "?":
        this.cleanedTargetName = value.substr(1);
        this.updateTarget = updateBooleanAttributeTarget;
        break;

      case "@":
        this.cleanedTargetName = value.substr(1);
        this.bind = triggerBind;
        this.unbind = triggerUnbind;
        break;

      default:
        this.cleanedTargetName = value;

        if (value === "class") {
          this.updateTarget = updateClassTarget;
        }

        break;
    }
  }
  /**
   * Makes this binding target the content of an element rather than
   * a particular attribute or property.
   */


  targetAtContent() {
    this.updateTarget = updateContentTarget;
    this.unbind = contentUnbind;
  }
  /**
   * Creates the runtime BindingBehavior instance based on the configuration
   * information stored in the BindingDirective.
   * @param target - The target node that the binding behavior should attach to.
   */


  createBehavior(target) {
    /* eslint-disable-next-line @typescript-eslint/no-use-before-define */
    return new BindingBehavior(target, this.binding, this.isBindingVolatile, this.bind, this.unbind, this.updateTarget, this.cleanedTargetName);
  }

}
/**
 * A behavior that updates content and attributes based on a configured
 * BindingDirective.
 * @public
 */

class BindingBehavior {
  /**
   * Creates an instance of BindingBehavior.
   * @param target - The target of the data updates.
   * @param binding - The binding that returns the latest value for an update.
   * @param isBindingVolatile - Indicates whether the binding has volatile dependencies.
   * @param bind - The operation to perform during binding.
   * @param unbind - The operation to perform during unbinding.
   * @param updateTarget - The operation to perform when updating.
   * @param targetName - The name of the target attribute or property to update.
   */
  constructor(target, binding, isBindingVolatile, bind, unbind, updateTarget, targetName) {
    /** @internal */
    this.source = null;
    /** @internal */

    this.context = null;
    /** @internal */

    this.bindingObserver = null;
    this.target = target;
    this.binding = binding;
    this.isBindingVolatile = isBindingVolatile;
    this.bind = bind;
    this.unbind = unbind;
    this.updateTarget = updateTarget;
    this.targetName = targetName;
  }
  /** @internal */


  handleChange() {
    this.updateTarget(this.bindingObserver.observe(this.source, this.context));
  }
  /** @internal */


  handleEvent(event) {
    ExecutionContext.setEvent(event);
    const result = this.binding(this.source, this.context);
    ExecutionContext.setEvent(null);

    if (result !== true) {
      event.preventDefault();
    }
  }

}

let sharedContext = null;

class CompilationContext {
  addFactory(factory) {
    factory.targetIndex = this.targetIndex;
    this.behaviorFactories.push(factory);
  }

  captureContentBinding(directive) {
    directive.targetAtContent();
    this.addFactory(directive);
  }

  reset() {
    this.behaviorFactories = [];
    this.targetIndex = -1;
  }

  release() {
    /* eslint-disable-next-line @typescript-eslint/no-this-alias */
    sharedContext = this;
  }

  static borrow(directives) {
    const shareable = sharedContext || new CompilationContext();
    shareable.directives = directives;
    shareable.reset();
    sharedContext = null;
    return shareable;
  }

}

function createAggregateBinding(parts) {
  if (parts.length === 1) {
    return parts[0];
  }

  let targetName;
  const partCount = parts.length;
  const finalParts = parts.map(x => {
    if (typeof x === "string") {
      return () => x;
    }

    targetName = x.targetName || targetName;
    return x.binding;
  });

  const binding = (scope, context) => {
    let output = "";

    for (let i = 0; i < partCount; ++i) {
      output += finalParts[i](scope, context);
    }

    return output;
  };

  const directive = new HTMLBindingDirective(binding);
  directive.targetName = targetName;
  return directive;
}

const interpolationEndLength = _interpolationEnd.length;

function parseContent(context, value) {
  const valueParts = value.split(_interpolationStart);

  if (valueParts.length === 1) {
    return null;
  }

  const bindingParts = [];

  for (let i = 0, ii = valueParts.length; i < ii; ++i) {
    const current = valueParts[i];
    const index = current.indexOf(_interpolationEnd);
    let literal;

    if (index === -1) {
      literal = current;
    } else {
      const directiveIndex = parseInt(current.substring(0, index));
      bindingParts.push(context.directives[directiveIndex]);
      literal = current.substring(index + interpolationEndLength);
    }

    if (literal !== "") {
      bindingParts.push(literal);
    }
  }

  return bindingParts;
}

function compileAttributes(context, node, includeBasicValues = false) {
  const attributes = node.attributes;

  for (let i = 0, ii = attributes.length; i < ii; ++i) {
    const attr = attributes[i];
    const attrValue = attr.value;
    const parseResult = parseContent(context, attrValue);
    let result = null;

    if (parseResult === null) {
      if (includeBasicValues) {
        result = new HTMLBindingDirective(() => attrValue);
        result.targetName = attr.name;
      }
    } else {
      result = createAggregateBinding(parseResult);
    }

    if (result !== null) {
      node.removeAttributeNode(attr);
      i--;
      ii--;
      context.addFactory(result);
    }
  }
}

function compileContent(context, node, walker) {
  const parseResult = parseContent(context, node.textContent);

  if (parseResult !== null) {
    let lastNode = node;

    for (let i = 0, ii = parseResult.length; i < ii; ++i) {
      const currentPart = parseResult[i];
      const currentNode = i === 0 ? node : lastNode.parentNode.insertBefore(document.createTextNode(""), lastNode.nextSibling);

      if (typeof currentPart === "string") {
        currentNode.textContent = currentPart;
      } else {
        currentNode.textContent = " ";
        context.captureContentBinding(currentPart);
      }

      lastNode = currentNode;
      context.targetIndex++;

      if (currentNode !== node) {
        walker.nextNode();
      }
    }

    context.targetIndex--;
  }
}
/**
 * Compiles a template and associated directives into a raw compilation
 * result which include a cloneable DocumentFragment and factories capable
 * of attaching runtime behavior to nodes within the fragment.
 * @param template - The template to compile.
 * @param directives - The directives referenced by the template.
 * @remarks
 * The template that is provided for compilation is altered in-place
 * and cannot be compiled again. If the original template must be preserved,
 * it is recommended that you clone the original and pass the clone to this API.
 * @public
 */


function compileTemplate(template, directives) {
  const fragment = template.content; // https://bugs.chromium.org/p/chromium/issues/detail?id=1111864

  document.adoptNode(fragment);
  const context = CompilationContext.borrow(directives);
  compileAttributes(context, template, true);
  const hostBehaviorFactories = context.behaviorFactories;
  context.reset();
  const walker = DOM.createTemplateWalker(fragment);
  let node;

  while (node = walker.nextNode()) {
    context.targetIndex++;

    switch (node.nodeType) {
      case 1:
        // element node
        compileAttributes(context, node);
        break;

      case 3:
        // text node
        compileContent(context, node, walker);
        break;

      case 8:
        // comment
        if (DOM.isMarker(node)) {
          context.addFactory(directives[DOM.extractDirectiveIndexFromMarker(node)]);
        }

    }
  }

  let targetOffset = 0;

  if ( // If the first node in a fragment is a marker, that means it's an unstable first node,
  // because something like a when, repeat, etc. could add nodes before the marker.
  // To mitigate this, we insert a stable first node. However, if we insert a node,
  // that will alter the result of the TreeWalker. So, we also need to offset the target index.
  DOM.isMarker(fragment.firstChild) || // Or if there is only one node and a directive, it means the template's content
  // is *only* the directive. In that case, HTMLView.dispose() misses any nodes inserted by
  // the directive. Inserting a new node ensures proper disposal of nodes added by the directive.
  fragment.childNodes.length === 1 && directives.length) {
    fragment.insertBefore(document.createComment(""), fragment.firstChild);
    targetOffset = -1;
  }

  const viewBehaviorFactories = context.behaviorFactories;
  context.release();
  return {
    fragment,
    viewBehaviorFactories,
    hostBehaviorFactories,
    targetOffset
  };
}

// A singleton Range instance used to efficiently remove ranges of DOM nodes.
// See the implementation of HTMLView below for further details.
const range = document.createRange();
/**
 * The standard View implementation, which also implements ElementView and SyntheticView.
 * @public
 */

class HTMLView {
  /**
   * Constructs an instance of HTMLView.
   * @param fragment - The html fragment that contains the nodes for this view.
   * @param behaviors - The behaviors to be applied to this view.
   */
  constructor(fragment, behaviors) {
    this.fragment = fragment;
    this.behaviors = behaviors;
    /**
     * The data that the view is bound to.
     */

    this.source = null;
    /**
     * The execution context the view is running within.
     */

    this.context = null;
    this.firstChild = fragment.firstChild;
    this.lastChild = fragment.lastChild;
  }
  /**
   * Appends the view's DOM nodes to the referenced node.
   * @param node - The parent node to append the view's DOM nodes to.
   */


  appendTo(node) {
    node.appendChild(this.fragment);
  }
  /**
   * Inserts the view's DOM nodes before the referenced node.
   * @param node - The node to insert the view's DOM before.
   */


  insertBefore(node) {
    if (this.fragment.hasChildNodes()) {
      node.parentNode.insertBefore(this.fragment, node);
    } else {
      const parentNode = node.parentNode;
      const end = this.lastChild;
      let current = this.firstChild;
      let next;

      while (current !== end) {
        next = current.nextSibling;
        parentNode.insertBefore(current, node);
        current = next;
      }

      parentNode.insertBefore(end, node);
    }
  }
  /**
   * Removes the view's DOM nodes.
   * The nodes are not disposed and the view can later be re-inserted.
   */


  remove() {
    const fragment = this.fragment;
    const end = this.lastChild;
    let current = this.firstChild;
    let next;

    while (current !== end) {
      next = current.nextSibling;
      fragment.appendChild(current);
      current = next;
    }

    fragment.appendChild(end);
  }
  /**
   * Removes the view and unbinds its behaviors, disposing of DOM nodes afterward.
   * Once a view has been disposed, it cannot be inserted or bound again.
   */


  dispose() {
    const parent = this.firstChild.parentNode;
    const end = this.lastChild;
    let current = this.firstChild;
    let next;

    while (current !== end) {
      next = current.nextSibling;
      parent.removeChild(current);
      current = next;
    }

    parent.removeChild(end);
    const behaviors = this.behaviors;
    const oldSource = this.source;

    for (let i = 0, ii = behaviors.length; i < ii; ++i) {
      behaviors[i].unbind(oldSource);
    }
  }
  /**
   * Binds a view's behaviors to its binding source.
   * @param source - The binding source for the view's binding behaviors.
   * @param context - The execution context to run the behaviors within.
   */


  bind(source, context) {
    const behaviors = this.behaviors;

    if (this.source === source) {
      return;
    } else if (this.source !== null) {
      const oldSource = this.source;
      this.source = source;
      this.context = context;

      for (let i = 0, ii = behaviors.length; i < ii; ++i) {
        const current = behaviors[i];
        current.unbind(oldSource);
        current.bind(source, context);
      }
    } else {
      this.source = source;
      this.context = context;

      for (let i = 0, ii = behaviors.length; i < ii; ++i) {
        behaviors[i].bind(source, context);
      }
    }
  }
  /**
   * Unbinds a view's behaviors from its binding source.
   */


  unbind() {
    if (this.source === null) {
      return;
    }

    const behaviors = this.behaviors;
    const oldSource = this.source;

    for (let i = 0, ii = behaviors.length; i < ii; ++i) {
      behaviors[i].unbind(oldSource);
    }

    this.source = null;
  }
  /**
   * Efficiently disposes of a contiguous range of synthetic view instances.
   * @param views - A contiguous range of views to be disposed.
   */


  static disposeContiguousBatch(views) {
    if (views.length === 0) {
      return;
    }

    range.setStartBefore(views[0].firstChild);
    range.setEndAfter(views[views.length - 1].lastChild);
    range.deleteContents();

    for (let i = 0, ii = views.length; i < ii; ++i) {
      const view = views[i];
      const behaviors = view.behaviors;
      const oldSource = view.source;

      for (let j = 0, jj = behaviors.length; j < jj; ++j) {
        behaviors[j].unbind(oldSource);
      }
    }
  }

}

/**
 * A template capable of creating HTMLView instances or rendering directly to DOM.
 * @public
 */

/* eslint-disable-next-line @typescript-eslint/no-unused-vars */

class ViewTemplate {
  /**
   * Creates an instance of ViewTemplate.
   * @param html - The html representing what this template will instantiate, including placeholders for directives.
   * @param directives - The directives that will be connected to placeholders in the html.
   */
  constructor(html, directives) {
    this.behaviorCount = 0;
    this.hasHostBehaviors = false;
    this.fragment = null;
    this.targetOffset = 0;
    this.viewBehaviorFactories = null;
    this.hostBehaviorFactories = null;
    this.html = html;
    this.directives = directives;
  }
  /**
   * Creates an HTMLView instance based on this template definition.
   * @param hostBindingTarget - The element that host behaviors will be bound to.
   */


  create(hostBindingTarget) {
    if (this.fragment === null) {
      let template;
      const html = this.html;

      if (typeof html === "string") {
        template = document.createElement("template");
        template.innerHTML = DOM.createHTML(html);
        const fec = template.content.firstElementChild;

        if (fec !== null && fec.tagName === "TEMPLATE") {
          template = fec;
        }
      } else {
        template = html;
      }

      const result = compileTemplate(template, this.directives);
      this.fragment = result.fragment;
      this.viewBehaviorFactories = result.viewBehaviorFactories;
      this.hostBehaviorFactories = result.hostBehaviorFactories;
      this.targetOffset = result.targetOffset;
      this.behaviorCount = this.viewBehaviorFactories.length + this.hostBehaviorFactories.length;
      this.hasHostBehaviors = this.hostBehaviorFactories.length > 0;
    }

    const fragment = this.fragment.cloneNode(true);
    const viewFactories = this.viewBehaviorFactories;
    const behaviors = new Array(this.behaviorCount);
    const walker = DOM.createTemplateWalker(fragment);
    let behaviorIndex = 0;
    let targetIndex = this.targetOffset;
    let node = walker.nextNode();

    for (let ii = viewFactories.length; behaviorIndex < ii; ++behaviorIndex) {
      const factory = viewFactories[behaviorIndex];
      const factoryIndex = factory.targetIndex;

      while (node !== null) {
        if (targetIndex === factoryIndex) {
          behaviors[behaviorIndex] = factory.createBehavior(node);
          break;
        } else {
          node = walker.nextNode();
          targetIndex++;
        }
      }
    }

    if (this.hasHostBehaviors) {
      const hostFactories = this.hostBehaviorFactories;

      for (let i = 0, ii = hostFactories.length; i < ii; ++i, ++behaviorIndex) {
        behaviors[behaviorIndex] = hostFactories[i].createBehavior(hostBindingTarget);
      }
    }

    return new HTMLView(fragment, behaviors);
  }
  /**
   * Creates an HTMLView from this template, binds it to the source, and then appends it to the host.
   * @param source - The data source to bind the template to.
   * @param host - The Element where the template will be rendered.
   * @param hostBindingTarget - An HTML element to target the host bindings at if different from the
   * host that the template is being attached to.
   */


  render(source, host, hostBindingTarget) {
    if (typeof host === "string") {
      host = document.getElementById(host);
    }

    if (hostBindingTarget === void 0) {
      hostBindingTarget = host;
    }

    const view = this.create(hostBindingTarget);
    view.bind(source, defaultExecutionContext);
    view.appendTo(host);
    return view;
  }

} // Much thanks to LitHTML for working this out!

const lastAttributeNameRegex =
/* eslint-disable-next-line no-control-regex */
/([ \x09\x0a\x0c\x0d])([^\0-\x1F\x7F-\x9F "'>=/]+)([ \x09\x0a\x0c\x0d]*=[ \x09\x0a\x0c\x0d]*(?:[^ \x09\x0a\x0c\x0d"'`<>=]*|"[^"]*|'[^']*))$/;
/**
 * Transforms a template literal string into a renderable ViewTemplate.
 * @param strings - The string fragments that are interpolated with the values.
 * @param values - The values that are interpolated with the string fragments.
 * @remarks
 * The html helper supports interpolation of strings, numbers, binding expressions,
 * other template instances, and Directive instances.
 * @public
 */

function html(strings, ...values) {
  const directives = [];
  let html = "";

  for (let i = 0, ii = strings.length - 1; i < ii; ++i) {
    const currentString = strings[i];
    let value = values[i];
    html += currentString;

    if (value instanceof ViewTemplate) {
      const template = value;

      value = () => template;
    }

    if (typeof value === "function") {
      value = new HTMLBindingDirective(value);
    }

    if (value instanceof TargetedHTMLDirective) {
      const match = lastAttributeNameRegex.exec(currentString);

      if (match !== null) {
        value.targetName = match[2];
      }
    }

    if (value instanceof HTMLDirective) {
      // Since not all values are directives, we can't use i
      // as the index for the placeholder. Instead, we need to
      // use directives.length to get the next index.
      html += value.createPlaceholder(directives.length);
      directives.push(value);
    } else {
      html += value;
    }
  }

  html += strings[strings.length - 1];
  return new ViewTemplate(html, directives);
}

/**
 * Represents styles that can be applied to a custom element.
 * @public
 */

class ElementStyles {
  constructor() {
    this.targets = new WeakSet();
  }
  /** @internal */


  addStylesTo(target) {
    this.targets.add(target);
  }
  /** @internal */


  removeStylesFrom(target) {
    this.targets.delete(target);
  }
  /** @internal */


  isAttachedTo(target) {
    return this.targets.has(target);
  }
  /**
   * Associates behaviors with this set of styles.
   * @param behaviors - The behaviors to associate.
   */


  withBehaviors(...behaviors) {
    this.behaviors = this.behaviors === null ? behaviors : this.behaviors.concat(behaviors);
    return this;
  }

}
/**
 * Create ElementStyles from ComposableStyles.
 */

ElementStyles.create = (() => {
  if (DOM.supportsAdoptedStyleSheets) {
    const styleSheetCache = new Map();
    return styles => // eslint-disable-next-line @typescript-eslint/no-use-before-define
    new AdoptedStyleSheetsStyles(styles, styleSheetCache);
  } // eslint-disable-next-line @typescript-eslint/no-use-before-define


  return styles => new StyleElementStyles(styles);
})();

function reduceStyles(styles) {
  return styles.map(x => x instanceof ElementStyles ? reduceStyles(x.styles) : [x]).reduce((prev, curr) => prev.concat(curr), []);
}

function reduceBehaviors(styles) {
  return styles.map(x => x instanceof ElementStyles ? x.behaviors : null).reduce((prev, curr) => {
    if (curr === null) {
      return prev;
    }

    if (prev === null) {
      prev = [];
    }

    return prev.concat(curr);
  }, null);
}
/**
 * https://wicg.github.io/construct-stylesheets/
 * https://developers.google.com/web/updates/2019/02/constructable-stylesheets
 *
 * @internal
 */


class AdoptedStyleSheetsStyles extends ElementStyles {
  constructor(styles, styleSheetCache) {
    super();
    this.styles = styles;
    this.styleSheetCache = styleSheetCache;
    this._styleSheets = void 0;
    this.behaviors = reduceBehaviors(styles);
  }

  get styleSheets() {
    if (this._styleSheets === void 0) {
      const styles = this.styles;
      const styleSheetCache = this.styleSheetCache;
      this._styleSheets = reduceStyles(styles).map(x => {
        if (x instanceof CSSStyleSheet) {
          return x;
        }

        let sheet = styleSheetCache.get(x);

        if (sheet === void 0) {
          sheet = new CSSStyleSheet();
          sheet.replaceSync(x);
          styleSheetCache.set(x, sheet);
        }

        return sheet;
      });
    }

    return this._styleSheets;
  }

  addStylesTo(target) {
    target.adoptedStyleSheets = [...target.adoptedStyleSheets, ...this.styleSheets];
    super.addStylesTo(target);
  }

  removeStylesFrom(target) {
    const sourceSheets = this.styleSheets;
    target.adoptedStyleSheets = target.adoptedStyleSheets.filter(x => sourceSheets.indexOf(x) === -1);
    super.removeStylesFrom(target);
  }

}
let styleClassId = 0;

function getNextStyleClass() {
  return `fast-style-class-${++styleClassId}`;
}
/**
 * @internal
 */


class StyleElementStyles extends ElementStyles {
  constructor(styles) {
    super();
    this.styles = styles;
    this.behaviors = null;
    this.behaviors = reduceBehaviors(styles);
    this.styleSheets = reduceStyles(styles);
    this.styleClass = getNextStyleClass();
  }

  addStylesTo(target) {
    const styleSheets = this.styleSheets;
    const styleClass = this.styleClass;
    target = this.normalizeTarget(target);

    for (let i = 0; i < styleSheets.length; i++) {
      const element = document.createElement("style");
      element.innerHTML = styleSheets[i];
      element.className = styleClass;
      target.append(element);
    }

    super.addStylesTo(target);
  }

  removeStylesFrom(target) {
    target = this.normalizeTarget(target);
    const styles = target.querySelectorAll(`.${this.styleClass}`);

    for (let i = 0, ii = styles.length; i < ii; ++i) {
      target.removeChild(styles[i]);
    }

    super.removeStylesFrom(target);
  }

  isAttachedTo(target) {
    return super.isAttachedTo(this.normalizeTarget(target));
  }

  normalizeTarget(target) {
    return target === document ? document.body : target;
  }

}

/**
 * A {@link ValueConverter} that converts to and from `boolean` values.
 * @remarks
 * Used automatically when the `boolean` {@link AttributeMode} is selected.
 * @public
 */

const booleanConverter = {
  toView(value) {
    return value ? "true" : "false";
  },

  fromView(value) {
    if (value === null || value === void 0 || value === "false" || value === false || value === 0) {
      return false;
    }

    return true;
  }

};
/**
 * A {@link ValueConverter} that converts to and from `number` values.
 * @remarks
 * This converter allows for nullable numbers, returning `null` if the
 * input was `null`, `undefined`, or `NaN`.
 * @public
 */

const nullableNumberConverter = {
  toView(value) {
    if (value === null || value === undefined) {
      return null;
    }

    const number = value * 1;
    return isNaN(number) ? null : number.toString();
  },

  fromView(value) {
    if (value === null || value === undefined) {
      return null;
    }

    const number = value * 1;
    return isNaN(number) ? null : number;
  }

};
/**
 * An implementation of {@link Accessor} that supports reactivity,
 * change callbacks, attribute reflection, and type conversion for
 * custom elements.
 * @public
 */

class AttributeDefinition {
  /**
   * Creates an instance of AttributeDefinition.
   * @param Owner - The class constructor that owns this attribute.
   * @param name - The name of the property associated with the attribute.
   * @param attribute - The name of the attribute in HTML.
   * @param mode - The {@link AttributeMode} that describes the behavior of this attribute.
   * @param converter - A {@link ValueConverter} that integrates with the property getter/setter
   * to convert values to and from a DOM string.
   */
  constructor(Owner, name, attribute = name.toLowerCase(), mode = "reflect", converter) {
    this.guards = new Set();
    this.Owner = Owner;
    this.name = name;
    this.attribute = attribute;
    this.mode = mode;
    this.converter = converter;
    this.fieldName = `_${name}`;
    this.callbackName = `${name}Changed`;
    this.hasCallback = this.callbackName in Owner.prototype;

    if (mode === "boolean" && converter === void 0) {
      this.converter = booleanConverter;
    }
  }
  /**
   * Sets the value of the attribute/property on the source element.
   * @param source - The source element to access.
   * @param value - The value to set the attribute/property to.
   */


  setValue(source, newValue) {
    const oldValue = source[this.fieldName];
    const converter = this.converter;

    if (converter !== void 0) {
      newValue = converter.fromView(newValue);
    }

    if (oldValue !== newValue) {
      source[this.fieldName] = newValue;
      this.tryReflectToAttribute(source);

      if (this.hasCallback) {
        source[this.callbackName](oldValue, newValue);
      }

      source.$fastController.notify(this.name);
    }
  }
  /**
   * Gets the value of the attribute/property on the source element.
   * @param source - The source element to access.
   */


  getValue(source) {
    Observable.track(source, this.name);
    return source[this.fieldName];
  }
  /** @internal */


  onAttributeChangedCallback(element, value) {
    if (this.guards.has(element)) {
      return;
    }

    this.guards.add(element);
    this.setValue(element, value);
    this.guards.delete(element);
  }

  tryReflectToAttribute(element) {
    const mode = this.mode;
    const guards = this.guards;

    if (guards.has(element) || mode === "fromView") {
      return;
    }

    DOM.queueUpdate(() => {
      guards.add(element);
      const latestValue = element[this.fieldName];

      switch (mode) {
        case "reflect":
          const converter = this.converter;
          DOM.setAttribute(element, this.attribute, converter !== void 0 ? converter.toView(latestValue) : latestValue);
          break;

        case "boolean":
          DOM.setBooleanAttribute(element, this.attribute, latestValue);
          break;
      }

      guards.delete(element);
    });
  }
  /**
   * Collects all attribute definitions associated with the owner.
   * @param Owner - The class constructor to collect attribute for.
   * @param attributeLists - Any existing attributes to collect and merge with those associated with the owner.
   * @internal
   */


  static collect(Owner, ...attributeLists) {
    const attributes = [];
    attributeLists.push(Owner.attributes);

    for (let i = 0, ii = attributeLists.length; i < ii; ++i) {
      const list = attributeLists[i];

      if (list === void 0) {
        continue;
      }

      for (let j = 0, jj = list.length; j < jj; ++j) {
        const config = list[j];

        if (typeof config === "string") {
          attributes.push(new AttributeDefinition(Owner, config));
        } else {
          attributes.push(new AttributeDefinition(Owner, config.property, config.attribute, config.mode, config.converter));
        }
      }
    }

    return attributes;
  }

}
function attr(configOrTarget, prop) {
  let config;

  function decorator($target, $prop) {
    if (arguments.length > 1) {
      // Non invocation:
      // - @attr
      // Invocation with or w/o opts:
      // - @attr()
      // - @attr({...opts})
      config.property = $prop;
    }

    const attributes = $target.constructor.attributes || ($target.constructor.attributes = []);
    attributes.push(config);
  }

  if (arguments.length > 1) {
    // Non invocation:
    // - @attr
    config = {};
    decorator(configOrTarget, prop);
    return;
  } // Invocation with or w/o opts:
  // - @attr()
  // - @attr({...opts})


  config = configOrTarget === void 0 ? {} : configOrTarget;
  return decorator;
}

const defaultShadowOptions = {
  mode: "open"
};
const defaultElementOptions = {};
const fastRegistry = FAST.getById(4
/* elementRegistry */
, () => {
  const typeToDefinition = new Map();
  return Object.freeze({
    register(definition) {
      if (typeToDefinition.has(definition.type)) {
        return false;
      }

      typeToDefinition.set(definition.type, definition);
      return true;
    },

    getByType(key) {
      return typeToDefinition.get(key);
    }

  });
});
/**
 * Defines metadata for a FASTElement.
 * @public
 */

class FASTElementDefinition {
  /**
   * Creates an instance of FASTElementDefinition.
   * @param type - The type this definition is being created for.
   * @param nameOrConfig - The name of the element to define or a config object
   * that describes the element to define.
   */
  constructor(type, nameOrConfig = type.definition) {
    if (typeof nameOrConfig === "string") {
      nameOrConfig = {
        name: nameOrConfig
      };
    }

    this.type = type;
    this.name = nameOrConfig.name;
    this.template = nameOrConfig.template;
    const attributes = AttributeDefinition.collect(type, nameOrConfig.attributes);
    const observedAttributes = new Array(attributes.length);
    const propertyLookup = {};
    const attributeLookup = {};

    for (let i = 0, ii = attributes.length; i < ii; ++i) {
      const current = attributes[i];
      observedAttributes[i] = current.attribute;
      propertyLookup[current.name] = current;
      attributeLookup[current.attribute] = current;
    }

    this.attributes = attributes;
    this.observedAttributes = observedAttributes;
    this.propertyLookup = propertyLookup;
    this.attributeLookup = attributeLookup;
    this.shadowOptions = nameOrConfig.shadowOptions === void 0 ? defaultShadowOptions : nameOrConfig.shadowOptions === null ? void 0 : Object.assign(Object.assign({}, defaultShadowOptions), nameOrConfig.shadowOptions);
    this.elementOptions = nameOrConfig.elementOptions === void 0 ? defaultElementOptions : Object.assign(Object.assign({}, defaultElementOptions), nameOrConfig.elementOptions);
    this.styles = nameOrConfig.styles === void 0 ? void 0 : Array.isArray(nameOrConfig.styles) ? ElementStyles.create(nameOrConfig.styles) : nameOrConfig.styles instanceof ElementStyles ? nameOrConfig.styles : ElementStyles.create([nameOrConfig.styles]);
  }
  /**
   * Indicates if this element has been defined in at least one registry.
   */


  get isDefined() {
    return !!fastRegistry.getByType(this.type);
  }
  /**
   * Defines a custom element based on this definition.
   * @param registry - The element registry to define the element in.
   */


  define(registry = customElements) {
    const type = this.type;

    if (fastRegistry.register(this)) {
      const attributes = this.attributes;
      const proto = type.prototype;

      for (let i = 0, ii = attributes.length; i < ii; ++i) {
        Observable.defineProperty(proto, attributes[i]);
      }

      Reflect.defineProperty(type, "observedAttributes", {
        value: this.observedAttributes,
        enumerable: true
      });
    }

    if (!registry.get(this.name)) {
      registry.define(this.name, type, this.elementOptions);
    }

    return this;
  }

}
/**
 * Gets the element definition associated with the specified type.
 * @param type - The custom element type to retrieve the definition for.
 */

FASTElementDefinition.forType = fastRegistry.getByType;

const shadowRoots = new WeakMap();
const defaultEventOptions = {
  bubbles: true,
  composed: true,
  cancelable: true
};

function getShadowRoot(element) {
  return element.shadowRoot || shadowRoots.get(element) || null;
}
/**
 * Controls the lifecycle and rendering of a `FASTElement`.
 * @public
 */


class Controller extends PropertyChangeNotifier {
  /**
   * Creates a Controller to control the specified element.
   * @param element - The element to be controlled by this controller.
   * @param definition - The element definition metadata that instructs this
   * controller in how to handle rendering and other platform integrations.
   * @internal
   */
  constructor(element, definition) {
    super(element);
    this.boundObservables = null;
    this.behaviors = null;
    this.needsInitialization = true;
    this._template = null;
    this._styles = null;
    this._isConnected = false;
    /**
     * This allows Observable.getNotifier(...) to return the Controller
     * when the notifier for the Controller itself is being requested. The
     * result is that the Observable system does not need to create a separate
     * instance of Notifier for observables on the Controller. The component and
     * the controller will now share the same notifier, removing one-object construct
     * per web component instance.
     */

    this.$fastController = this;
    /**
     * The view associated with the custom element.
     * @remarks
     * If `null` then the element is managing its own rendering.
     */

    this.view = null;
    this.element = element;
    this.definition = definition;
    const shadowOptions = definition.shadowOptions;

    if (shadowOptions !== void 0) {
      const shadowRoot = element.attachShadow(shadowOptions);

      if (shadowOptions.mode === "closed") {
        shadowRoots.set(element, shadowRoot);
      }
    } // Capture any observable values that were set by the binding engine before
    // the browser upgraded the element. Then delete the property since it will
    // shadow the getter/setter that is required to make the observable operate.
    // Later, in the connect callback, we'll re-apply the values.


    const accessors = Observable.getAccessors(element);

    if (accessors.length > 0) {
      const boundObservables = this.boundObservables = Object.create(null);

      for (let i = 0, ii = accessors.length; i < ii; ++i) {
        const propertyName = accessors[i].name;
        const value = element[propertyName];

        if (value !== void 0) {
          delete element[propertyName];
          boundObservables[propertyName] = value;
        }
      }
    }
  }
  /**
   * Indicates whether or not the custom element has been
   * connected to the document.
   */


  get isConnected() {
    Observable.track(this, "isConnected");
    return this._isConnected;
  }

  setIsConnected(value) {
    this._isConnected = value;
    Observable.notify(this, "isConnected");
  }
  /**
   * Gets/sets the template used to render the component.
   * @remarks
   * This value can only be accurately read after connect but can be set at any time.
   */


  get template() {
    return this._template;
  }

  set template(value) {
    if (this._template === value) {
      return;
    }

    this._template = value;

    if (!this.needsInitialization) {
      this.renderTemplate(value);
    }
  }
  /**
   * Gets/sets the primary styles used for the component.
   * @remarks
   * This value can only be accurately read after connect but can be set at any time.
   */


  get styles() {
    return this._styles;
  }

  set styles(value) {
    if (this._styles === value) {
      return;
    }

    if (this._styles !== null) {
      this.removeStyles(this._styles);
    }

    this._styles = value;

    if (!this.needsInitialization && value !== null) {
      this.addStyles(value);
    }
  }
  /**
   * Adds styles to this element. Providing an HTMLStyleElement will attach the element instance to the shadowRoot.
   * @param styles - The styles to add.
   */


  addStyles(styles) {
    const target = getShadowRoot(this.element) || this.element.getRootNode();

    if (styles instanceof HTMLStyleElement) {
      target.append(styles);
    } else if (!styles.isAttachedTo(target)) {
      const sourceBehaviors = styles.behaviors;
      styles.addStylesTo(target);

      if (sourceBehaviors !== null) {
        this.addBehaviors(sourceBehaviors);
      }
    }
  }
  /**
   * Removes styles from this element. Providing an HTMLStyleElement will detach the element instance from the shadowRoot.
   * @param styles - the styles to remove.
   */


  removeStyles(styles) {
    const target = getShadowRoot(this.element) || this.element.getRootNode();

    if (styles instanceof HTMLStyleElement) {
      target.removeChild(styles);
    } else if (styles.isAttachedTo(target)) {
      const sourceBehaviors = styles.behaviors;
      styles.removeStylesFrom(target);

      if (sourceBehaviors !== null) {
        this.removeBehaviors(sourceBehaviors);
      }
    }
  }
  /**
   * Adds behaviors to this element.
   * @param behaviors - The behaviors to add.
   */


  addBehaviors(behaviors) {
    const targetBehaviors = this.behaviors || (this.behaviors = new Map());
    const length = behaviors.length;
    const behaviorsToBind = [];

    for (let i = 0; i < length; ++i) {
      const behavior = behaviors[i];

      if (targetBehaviors.has(behavior)) {
        targetBehaviors.set(behavior, targetBehaviors.get(behavior) + 1);
      } else {
        targetBehaviors.set(behavior, 1);
        behaviorsToBind.push(behavior);
      }
    }

    if (this._isConnected) {
      const element = this.element;

      for (let i = 0; i < behaviorsToBind.length; ++i) {
        behaviorsToBind[i].bind(element, defaultExecutionContext);
      }
    }
  }
  /**
   * Removes behaviors from this element.
   * @param behaviors - The behaviors to remove.
   * @param force - Forces unbinding of behaviors.
   */


  removeBehaviors(behaviors, force = false) {
    const targetBehaviors = this.behaviors;

    if (targetBehaviors === null) {
      return;
    }

    const length = behaviors.length;
    const behaviorsToUnbind = [];

    for (let i = 0; i < length; ++i) {
      const behavior = behaviors[i];

      if (targetBehaviors.has(behavior)) {
        const count = targetBehaviors.get(behavior) - 1;
        count === 0 || force ? targetBehaviors.delete(behavior) && behaviorsToUnbind.push(behavior) : targetBehaviors.set(behavior, count);
      }
    }

    if (this._isConnected) {
      const element = this.element;

      for (let i = 0; i < behaviorsToUnbind.length; ++i) {
        behaviorsToUnbind[i].unbind(element);
      }
    }
  }
  /**
   * Runs connected lifecycle behavior on the associated element.
   */


  onConnectedCallback() {
    if (this._isConnected) {
      return;
    }

    const element = this.element;

    if (this.needsInitialization) {
      this.finishInitialization();
    } else if (this.view !== null) {
      this.view.bind(element, defaultExecutionContext);
    }

    const behaviors = this.behaviors;

    if (behaviors !== null) {
      for (const [behavior] of behaviors) {
        behavior.bind(element, defaultExecutionContext);
      }
    }

    this.setIsConnected(true);
  }
  /**
   * Runs disconnected lifecycle behavior on the associated element.
   */


  onDisconnectedCallback() {
    if (!this._isConnected) {
      return;
    }

    this.setIsConnected(false);
    const view = this.view;

    if (view !== null) {
      view.unbind();
    }

    const behaviors = this.behaviors;

    if (behaviors !== null) {
      const element = this.element;

      for (const [behavior] of behaviors) {
        behavior.unbind(element);
      }
    }
  }
  /**
   * Runs the attribute changed callback for the associated element.
   * @param name - The name of the attribute that changed.
   * @param oldValue - The previous value of the attribute.
   * @param newValue - The new value of the attribute.
   */


  onAttributeChangedCallback(name, oldValue, newValue) {
    const attrDef = this.definition.attributeLookup[name];

    if (attrDef !== void 0) {
      attrDef.onAttributeChangedCallback(this.element, newValue);
    }
  }
  /**
   * Emits a custom HTML event.
   * @param type - The type name of the event.
   * @param detail - The event detail object to send with the event.
   * @param options - The event options. By default bubbles and composed.
   * @remarks
   * Only emits events if connected.
   */


  emit(type, detail, options) {
    if (this._isConnected) {
      return this.element.dispatchEvent(new CustomEvent(type, Object.assign(Object.assign({
        detail
      }, defaultEventOptions), options)));
    }

    return false;
  }

  finishInitialization() {
    const element = this.element;
    const boundObservables = this.boundObservables; // If we have any observables that were bound, re-apply their values.

    if (boundObservables !== null) {
      const propertyNames = Object.keys(boundObservables);

      for (let i = 0, ii = propertyNames.length; i < ii; ++i) {
        const propertyName = propertyNames[i];
        element[propertyName] = boundObservables[propertyName];
      }

      this.boundObservables = null;
    }

    const definition = this.definition; // 1. Template overrides take top precedence.

    if (this._template === null) {
      if (this.element.resolveTemplate) {
        // 2. Allow for element instance overrides next.
        this._template = this.element.resolveTemplate();
      } else if (definition.template) {
        // 3. Default to the static definition.
        this._template = definition.template || null;
      }
    } // If we have a template after the above process, render it.
    // If there's no template, then the element author has opted into
    // custom rendering and they will managed the shadow root's content themselves.


    if (this._template !== null) {
      this.renderTemplate(this._template);
    } // 1. Styles overrides take top precedence.


    if (this._styles === null) {
      if (this.element.resolveStyles) {
        // 2. Allow for element instance overrides next.
        this._styles = this.element.resolveStyles();
      } else if (definition.styles) {
        // 3. Default to the static definition.
        this._styles = definition.styles || null;
      }
    } // If we have styles after the above process, add them.


    if (this._styles !== null) {
      this.addStyles(this._styles);
    }

    this.needsInitialization = false;
  }

  renderTemplate(template) {
    const element = this.element; // When getting the host to render to, we start by looking
    // up the shadow root. If there isn't one, then that means
    // we're doing a Light DOM render to the element's direct children.

    const host = getShadowRoot(element) || element;

    if (this.view !== null) {
      // If there's already a view, we need to unbind and remove through dispose.
      this.view.dispose();
      this.view = null;
    } else if (!this.needsInitialization) {
      // If there was previous custom rendering, we need to clear out the host.
      DOM.removeChildNodes(host);
    }

    if (template) {
      // If a new template was provided, render it.
      this.view = template.render(element, host, element);
    }
  }
  /**
   * Locates or creates a controller for the specified element.
   * @param element - The element to return the controller for.
   * @remarks
   * The specified element must have a {@link FASTElementDefinition}
   * registered either through the use of the {@link customElement}
   * decorator or a call to `FASTElement.define`.
   */


  static forCustomElement(element) {
    const controller = element.$fastController;

    if (controller !== void 0) {
      return controller;
    }

    const definition = FASTElementDefinition.forType(element.constructor);

    if (definition === void 0) {
      throw new Error("Missing FASTElement definition.");
    }

    return element.$fastController = new Controller(element, definition);
  }

}

/* eslint-disable-next-line @typescript-eslint/explicit-function-return-type */

function createFASTElement(BaseType) {
  return class extends BaseType {
    constructor() {
      /* eslint-disable-next-line */
      super();
      Controller.forCustomElement(this);
    }

    $emit(type, detail, options) {
      return this.$fastController.emit(type, detail, options);
    }

    connectedCallback() {
      this.$fastController.onConnectedCallback();
    }

    disconnectedCallback() {
      this.$fastController.onDisconnectedCallback();
    }

    attributeChangedCallback(name, oldValue, newValue) {
      this.$fastController.onAttributeChangedCallback(name, oldValue, newValue);
    }

  };
}
/**
 * A minimal base class for FASTElements that also provides
 * static helpers for working with FASTElements.
 * @public
 */


const FASTElement = Object.assign(createFASTElement(HTMLElement), {
  /**
   * Creates a new FASTElement base class inherited from the
   * provided base type.
   * @param BaseType - The base element type to inherit from.
   */
  from(BaseType) {
    return createFASTElement(BaseType);
  },

  /**
   * Defines a platform custom element based on the provided type and definition.
   * @param type - The custom element type to define.
   * @param nameOrDef - The name of the element to define or a definition object
   * that describes the element to define.
   */
  define(type, nameOrDef) {
    return new FASTElementDefinition(type, nameOrDef).define().type;
  }

});

/**
 * Directive for use in {@link css}.
 *
 * @public
 */
class CSSDirective {
  /**
   * Creates a CSS fragment to interpolate into the CSS document.
   * @returns - the string to interpolate into CSS
   */
  createCSS() {
    return "";
  }
  /**
   * Creates a behavior to bind to the host element.
   * @returns - the behavior to bind to the host element, or undefined.
   */


  createBehavior() {
    return undefined;
  }

}

function collectStyles(strings, values) {
  const styles = [];
  let cssString = "";
  const behaviors = [];

  for (let i = 0, ii = strings.length - 1; i < ii; ++i) {
    cssString += strings[i];
    let value = values[i];

    if (value instanceof CSSDirective) {
      const behavior = value.createBehavior();
      value = value.createCSS();

      if (behavior) {
        behaviors.push(behavior);
      }
    }

    if (value instanceof ElementStyles || value instanceof CSSStyleSheet) {
      if (cssString.trim() !== "") {
        styles.push(cssString);
        cssString = "";
      }

      styles.push(value);
    } else {
      cssString += value;
    }
  }

  cssString += strings[strings.length - 1];

  if (cssString.trim() !== "") {
    styles.push(cssString);
  }

  return {
    styles,
    behaviors
  };
}
/**
 * Transforms a template literal string into styles.
 * @param strings - The string fragments that are interpolated with the values.
 * @param values - The values that are interpolated with the string fragments.
 * @remarks
 * The css helper supports interpolation of strings and ElementStyle instances.
 * @public
 */


function css(strings, ...values) {
  const {
    styles,
    behaviors
  } = collectStyles(strings, values);
  const elementStyles = ElementStyles.create(styles);

  if (behaviors.length) {
    elementStyles.withBehaviors(...behaviors);
  }

  return elementStyles;
}

class CSSPartial extends CSSDirective {
  constructor(styles, behaviors) {
    super();
    this.behaviors = behaviors;
    this.css = "";
    const stylesheets = styles.reduce((accumulated, current) => {
      if (typeof current === "string") {
        this.css += current;
      } else {
        accumulated.push(current);
      }

      return accumulated;
    }, []);

    if (stylesheets.length) {
      this.styles = ElementStyles.create(stylesheets);
    }
  }

  createBehavior() {
    return this;
  }

  createCSS() {
    return this.css;
  }

  bind(el) {
    if (this.styles) {
      el.$fastController.addStyles(this.styles);
    }

    if (this.behaviors.length) {
      el.$fastController.addBehaviors(this.behaviors);
    }
  }

  unbind(el) {
    if (this.styles) {
      el.$fastController.removeStyles(this.styles);
    }

    if (this.behaviors.length) {
      el.$fastController.removeBehaviors(this.behaviors);
    }
  }

}
/**
 * Transforms a template literal string into partial CSS.
 * @param strings - The string fragments that are interpolated with the values.
 * @param values - The values that are interpolated with the string fragments.
 * @public
 */


function cssPartial(strings, ...values) {
  const {
    styles,
    behaviors
  } = collectStyles(strings, values);
  return new CSSPartial(styles, behaviors);
}

/** @internal */

function newSplice(index, removed, addedCount) {
  return {
    index: index,
    removed: removed,
    addedCount: addedCount
  };
}
const EDIT_LEAVE = 0;
const EDIT_UPDATE = 1;
const EDIT_ADD = 2;
const EDIT_DELETE = 3; // Note: This function is *based* on the computation of the Levenshtein
// "edit" distance. The one change is that "updates" are treated as two
// edits - not one. With Array splices, an update is really a delete
// followed by an add. By retaining this, we optimize for "keeping" the
// maximum array items in the original array. For example:
//
//   'xxxx123' -> '123yyyy'
//
// With 1-edit updates, the shortest path would be just to update all seven
// characters. With 2-edit updates, we delete 4, leave 3, and add 4. This
// leaves the substring '123' intact.

function calcEditDistances(current, currentStart, currentEnd, old, oldStart, oldEnd) {
  // "Deletion" columns
  const rowCount = oldEnd - oldStart + 1;
  const columnCount = currentEnd - currentStart + 1;
  const distances = new Array(rowCount);
  let north;
  let west; // "Addition" rows. Initialize null column.

  for (let i = 0; i < rowCount; ++i) {
    distances[i] = new Array(columnCount);
    distances[i][0] = i;
  } // Initialize null row


  for (let j = 0; j < columnCount; ++j) {
    distances[0][j] = j;
  }

  for (let i = 1; i < rowCount; ++i) {
    for (let j = 1; j < columnCount; ++j) {
      if (current[currentStart + j - 1] === old[oldStart + i - 1]) {
        distances[i][j] = distances[i - 1][j - 1];
      } else {
        north = distances[i - 1][j] + 1;
        west = distances[i][j - 1] + 1;
        distances[i][j] = north < west ? north : west;
      }
    }
  }

  return distances;
} // This starts at the final weight, and walks "backward" by finding
// the minimum previous weight recursively until the origin of the weight
// matrix.


function spliceOperationsFromEditDistances(distances) {
  let i = distances.length - 1;
  let j = distances[0].length - 1;
  let current = distances[i][j];
  const edits = [];

  while (i > 0 || j > 0) {
    if (i === 0) {
      edits.push(EDIT_ADD);
      j--;
      continue;
    }

    if (j === 0) {
      edits.push(EDIT_DELETE);
      i--;
      continue;
    }

    const northWest = distances[i - 1][j - 1];
    const west = distances[i - 1][j];
    const north = distances[i][j - 1];
    let min;

    if (west < north) {
      min = west < northWest ? west : northWest;
    } else {
      min = north < northWest ? north : northWest;
    }

    if (min === northWest) {
      if (northWest === current) {
        edits.push(EDIT_LEAVE);
      } else {
        edits.push(EDIT_UPDATE);
        current = northWest;
      }

      i--;
      j--;
    } else if (min === west) {
      edits.push(EDIT_DELETE);
      i--;
      current = west;
    } else {
      edits.push(EDIT_ADD);
      j--;
      current = north;
    }
  }

  edits.reverse();
  return edits;
}

function sharedPrefix(current, old, searchLength) {
  for (let i = 0; i < searchLength; ++i) {
    if (current[i] !== old[i]) {
      return i;
    }
  }

  return searchLength;
}

function sharedSuffix(current, old, searchLength) {
  let index1 = current.length;
  let index2 = old.length;
  let count = 0;

  while (count < searchLength && current[--index1] === old[--index2]) {
    count++;
  }

  return count;
}

function intersect(start1, end1, start2, end2) {
  // Disjoint
  if (end1 < start2 || end2 < start1) {
    return -1;
  } // Adjacent


  if (end1 === start2 || end2 === start1) {
    return 0;
  } // Non-zero intersect, span1 first


  if (start1 < start2) {
    if (end1 < end2) {
      return end1 - start2; // Overlap
    }

    return end2 - start2; // Contained
  } // Non-zero intersect, span2 first


  if (end2 < end1) {
    return end2 - start1; // Overlap
  }

  return end1 - start1; // Contained
}
/**
 * Splice Projection functions:
 *
 * A splice map is a representation of how a previous array of items
 * was transformed into a new array of items. Conceptually it is a list of
 * tuples of
 *
 *   <index, removed, addedCount>
 *
 * which are kept in ascending index order of. The tuple represents that at
 * the |index|, |removed| sequence of items were removed, and counting forward
 * from |index|, |addedCount| items were added.
 */

/**
 * @internal
 * @remarks
 * Lacking individual splice mutation information, the minimal set of
 * splices can be synthesized given the previous state and final state of an
 * array. The basic approach is to calculate the edit distance matrix and
 * choose the shortest path through it.
 *
 * Complexity: O(l * p)
 *   l: The length of the current array
 *   p: The length of the old array
 */


function calcSplices(current, currentStart, currentEnd, old, oldStart, oldEnd) {
  let prefixCount = 0;
  let suffixCount = 0;
  const minLength = Math.min(currentEnd - currentStart, oldEnd - oldStart);

  if (currentStart === 0 && oldStart === 0) {
    prefixCount = sharedPrefix(current, old, minLength);
  }

  if (currentEnd === current.length && oldEnd === old.length) {
    suffixCount = sharedSuffix(current, old, minLength - prefixCount);
  }

  currentStart += prefixCount;
  oldStart += prefixCount;
  currentEnd -= suffixCount;
  oldEnd -= suffixCount;

  if (currentEnd - currentStart === 0 && oldEnd - oldStart === 0) {
    return emptyArray;
  }

  if (currentStart === currentEnd) {
    const splice = newSplice(currentStart, [], 0);

    while (oldStart < oldEnd) {
      splice.removed.push(old[oldStart++]);
    }

    return [splice];
  } else if (oldStart === oldEnd) {
    return [newSplice(currentStart, [], currentEnd - currentStart)];
  }

  const ops = spliceOperationsFromEditDistances(calcEditDistances(current, currentStart, currentEnd, old, oldStart, oldEnd));
  const splices = [];
  let splice = void 0;
  let index = currentStart;
  let oldIndex = oldStart;

  for (let i = 0; i < ops.length; ++i) {
    switch (ops[i]) {
      case EDIT_LEAVE:
        if (splice !== void 0) {
          splices.push(splice);
          splice = void 0;
        }

        index++;
        oldIndex++;
        break;

      case EDIT_UPDATE:
        if (splice === void 0) {
          splice = newSplice(index, [], 0);
        }

        splice.addedCount++;
        index++;
        splice.removed.push(old[oldIndex]);
        oldIndex++;
        break;

      case EDIT_ADD:
        if (splice === void 0) {
          splice = newSplice(index, [], 0);
        }

        splice.addedCount++;
        index++;
        break;

      case EDIT_DELETE:
        if (splice === void 0) {
          splice = newSplice(index, [], 0);
        }

        splice.removed.push(old[oldIndex]);
        oldIndex++;
        break;
      // no default
    }
  }

  if (splice !== void 0) {
    splices.push(splice);
  }

  return splices;
}
const $push = Array.prototype.push;

function mergeSplice(splices, index, removed, addedCount) {
  const splice = newSplice(index, removed, addedCount);
  let inserted = false;
  let insertionOffset = 0;

  for (let i = 0; i < splices.length; i++) {
    const current = splices[i];
    current.index += insertionOffset;

    if (inserted) {
      continue;
    }

    const intersectCount = intersect(splice.index, splice.index + splice.removed.length, current.index, current.index + current.addedCount);

    if (intersectCount >= 0) {
      // Merge the two splices
      splices.splice(i, 1);
      i--;
      insertionOffset -= current.addedCount - current.removed.length;
      splice.addedCount += current.addedCount - intersectCount;
      const deleteCount = splice.removed.length + current.removed.length - intersectCount;

      if (!splice.addedCount && !deleteCount) {
        // merged splice is a noop. discard.
        inserted = true;
      } else {
        let currentRemoved = current.removed;

        if (splice.index < current.index) {
          // some prefix of splice.removed is prepended to current.removed.
          const prepend = splice.removed.slice(0, current.index - splice.index);
          $push.apply(prepend, currentRemoved);
          currentRemoved = prepend;
        }

        if (splice.index + splice.removed.length > current.index + current.addedCount) {
          // some suffix of splice.removed is appended to current.removed.
          const append = splice.removed.slice(current.index + current.addedCount - splice.index);
          $push.apply(currentRemoved, append);
        }

        splice.removed = currentRemoved;

        if (current.index < splice.index) {
          splice.index = current.index;
        }
      }
    } else if (splice.index < current.index) {
      // Insert splice here.
      inserted = true;
      splices.splice(i, 0, splice);
      i++;
      const offset = splice.addedCount - splice.removed.length;
      current.index += offset;
      insertionOffset += offset;
    }
  }

  if (!inserted) {
    splices.push(splice);
  }
}

function createInitialSplices(changeRecords) {
  const splices = [];

  for (let i = 0, ii = changeRecords.length; i < ii; i++) {
    const record = changeRecords[i];
    mergeSplice(splices, record.index, record.removed, record.addedCount);
  }

  return splices;
}
/** @internal */


function projectArraySplices(array, changeRecords) {
  let splices = [];
  const initialSplices = createInitialSplices(changeRecords);

  for (let i = 0, ii = initialSplices.length; i < ii; ++i) {
    const splice = initialSplices[i];

    if (splice.addedCount === 1 && splice.removed.length === 1) {
      if (splice.removed[0] !== array[splice.index]) {
        splices.push(splice);
      }

      continue;
    }

    splices = splices.concat(calcSplices(array, splice.index, splice.index + splice.addedCount, splice.removed, 0, splice.removed.length));
  }

  return splices;
}

let arrayObservationEnabled = false;

function adjustIndex(changeRecord, array) {
  let index = changeRecord.index;
  const arrayLength = array.length;

  if (index > arrayLength) {
    index = arrayLength - changeRecord.addedCount;
  } else if (index < 0) {
    index = arrayLength + changeRecord.removed.length + index - changeRecord.addedCount;
  }

  if (index < 0) {
    index = 0;
  }

  changeRecord.index = index;
  return changeRecord;
}

class ArrayObserver extends SubscriberSet {
  constructor(source) {
    super(source);
    this.oldCollection = void 0;
    this.splices = void 0;
    this.needsQueue = true;
    this.call = this.flush;
    Reflect.defineProperty(source, "$fastController", {
      value: this,
      enumerable: false
    });
  }

  addSplice(splice) {
    if (this.splices === void 0) {
      this.splices = [splice];
    } else {
      this.splices.push(splice);
    }

    if (this.needsQueue) {
      this.needsQueue = false;
      DOM.queueUpdate(this);
    }
  }

  reset(oldCollection) {
    this.oldCollection = oldCollection;

    if (this.needsQueue) {
      this.needsQueue = false;
      DOM.queueUpdate(this);
    }
  }

  flush() {
    const splices = this.splices;
    const oldCollection = this.oldCollection;

    if (splices === void 0 && oldCollection === void 0) {
      return;
    }

    this.needsQueue = true;
    this.splices = void 0;
    this.oldCollection = void 0;
    const finalSplices = oldCollection === void 0 ? projectArraySplices(this.source, splices) : calcSplices(this.source, 0, this.source.length, oldCollection, 0, oldCollection.length);
    this.notify(finalSplices);
  }

}
/* eslint-disable prefer-rest-params */

/* eslint-disable @typescript-eslint/explicit-function-return-type */

/**
 * Enables the array observation mechanism.
 * @remarks
 * Array observation is enabled automatically when using the
 * {@link RepeatDirective}, so calling this API manually is
 * not typically necessary.
 * @public
 */


function enableArrayObservation() {
  if (arrayObservationEnabled) {
    return;
  }

  arrayObservationEnabled = true;
  Observable.setArrayObserverFactory(collection => {
    return new ArrayObserver(collection);
  });
  const proto = Array.prototype; // Don't patch Array if it has already been patched
  // by another copy of fast-element.

  if (proto.$fastPatch) {
    return;
  }

  Reflect.defineProperty(proto, "$fastPatch", {
    value: 1,
    enumerable: false
  });
  const pop = proto.pop;
  const push = proto.push;
  const reverse = proto.reverse;
  const shift = proto.shift;
  const sort = proto.sort;
  const splice = proto.splice;
  const unshift = proto.unshift;

  proto.pop = function () {
    const notEmpty = this.length > 0;
    const methodCallResult = pop.apply(this, arguments);
    const o = this.$fastController;

    if (o !== void 0 && notEmpty) {
      o.addSplice(newSplice(this.length, [methodCallResult], 0));
    }

    return methodCallResult;
  };

  proto.push = function () {
    const methodCallResult = push.apply(this, arguments);
    const o = this.$fastController;

    if (o !== void 0) {
      o.addSplice(adjustIndex(newSplice(this.length - arguments.length, [], arguments.length), this));
    }

    return methodCallResult;
  };

  proto.reverse = function () {
    let oldArray;
    const o = this.$fastController;

    if (o !== void 0) {
      o.flush();
      oldArray = this.slice();
    }

    const methodCallResult = reverse.apply(this, arguments);

    if (o !== void 0) {
      o.reset(oldArray);
    }

    return methodCallResult;
  };

  proto.shift = function () {
    const notEmpty = this.length > 0;
    const methodCallResult = shift.apply(this, arguments);
    const o = this.$fastController;

    if (o !== void 0 && notEmpty) {
      o.addSplice(newSplice(0, [methodCallResult], 0));
    }

    return methodCallResult;
  };

  proto.sort = function () {
    let oldArray;
    const o = this.$fastController;

    if (o !== void 0) {
      o.flush();
      oldArray = this.slice();
    }

    const methodCallResult = sort.apply(this, arguments);

    if (o !== void 0) {
      o.reset(oldArray);
    }

    return methodCallResult;
  };

  proto.splice = function () {
    const methodCallResult = splice.apply(this, arguments);
    const o = this.$fastController;

    if (o !== void 0) {
      o.addSplice(adjustIndex(newSplice(+arguments[0], methodCallResult, arguments.length > 2 ? arguments.length - 2 : 0), this));
    }

    return methodCallResult;
  };

  proto.unshift = function () {
    const methodCallResult = unshift.apply(this, arguments);
    const o = this.$fastController;

    if (o !== void 0) {
      o.addSplice(adjustIndex(newSplice(0, [], arguments.length), this));
    }

    return methodCallResult;
  };
}
/* eslint-enable prefer-rest-params */

/* eslint-enable @typescript-eslint/explicit-function-return-type */

/**
 * The runtime behavior for template references.
 * @public
 */

class RefBehavior {
  /**
   * Creates an instance of RefBehavior.
   * @param target - The element to reference.
   * @param propertyName - The name of the property to assign the reference to.
   */
  constructor(target, propertyName) {
    this.target = target;
    this.propertyName = propertyName;
  }
  /**
   * Bind this behavior to the source.
   * @param source - The source to bind to.
   * @param context - The execution context that the binding is operating within.
   */


  bind(source) {
    source[this.propertyName] = this.target;
  }
  /**
   * Unbinds this behavior from the source.
   * @param source - The source to unbind from.
   */

  /* eslint-disable-next-line @typescript-eslint/no-empty-function */


  unbind() {}

}
/**
 * A directive that observes the updates a property with a reference to the element.
 * @param propertyName - The name of the property to assign the reference to.
 * @public
 */

function ref(propertyName) {
  return new AttachedBehaviorHTMLDirective("fast-ref", RefBehavior, propertyName);
}

/**
 * A directive that enables basic conditional rendering in a template.
 * @param binding - The condition to test for rendering.
 * @param templateOrTemplateBinding - The template or a binding that gets
 * the template to render when the condition is true.
 * @public
 */
function when(binding, templateOrTemplateBinding) {
  const getTemplate = typeof templateOrTemplateBinding === "function" ? templateOrTemplateBinding : () => templateOrTemplateBinding;
  return (source, context) => binding(source, context) ? getTemplate(source, context) : null;
}

const defaultRepeatOptions = Object.freeze({
  positioning: false,
  recycle: true
});

function bindWithoutPositioning(view, items, index, context) {
  view.bind(items[index], context);
}

function bindWithPositioning(view, items, index, context) {
  const childContext = Object.create(context);
  childContext.index = index;
  childContext.length = items.length;
  view.bind(items[index], childContext);
}
/**
 * A behavior that renders a template for each item in an array.
 * @public
 */


class RepeatBehavior {
  /**
   * Creates an instance of RepeatBehavior.
   * @param location - The location in the DOM to render the repeat.
   * @param itemsBinding - The array to render.
   * @param isItemsBindingVolatile - Indicates whether the items binding has volatile dependencies.
   * @param templateBinding - The template to render for each item.
   * @param isTemplateBindingVolatile - Indicates whether the template binding has volatile dependencies.
   * @param options - Options used to turn on special repeat features.
   */
  constructor(location, itemsBinding, isItemsBindingVolatile, templateBinding, isTemplateBindingVolatile, options) {
    this.location = location;
    this.itemsBinding = itemsBinding;
    this.templateBinding = templateBinding;
    this.options = options;
    this.source = null;
    this.views = [];
    this.items = null;
    this.itemsObserver = null;
    this.originalContext = void 0;
    this.childContext = void 0;
    this.bindView = bindWithoutPositioning;
    this.itemsBindingObserver = Observable.binding(itemsBinding, this, isItemsBindingVolatile);
    this.templateBindingObserver = Observable.binding(templateBinding, this, isTemplateBindingVolatile);

    if (options.positioning) {
      this.bindView = bindWithPositioning;
    }
  }
  /**
   * Bind this behavior to the source.
   * @param source - The source to bind to.
   * @param context - The execution context that the binding is operating within.
   */


  bind(source, context) {
    this.source = source;
    this.originalContext = context;
    this.childContext = Object.create(context);
    this.childContext.parent = source;
    this.childContext.parentContext = this.originalContext;
    this.items = this.itemsBindingObserver.observe(source, this.originalContext);
    this.template = this.templateBindingObserver.observe(source, this.originalContext);
    this.observeItems(true);
    this.refreshAllViews();
  }
  /**
   * Unbinds this behavior from the source.
   * @param source - The source to unbind from.
   */


  unbind() {
    this.source = null;
    this.items = null;

    if (this.itemsObserver !== null) {
      this.itemsObserver.unsubscribe(this);
    }

    this.unbindAllViews();
    this.itemsBindingObserver.disconnect();
    this.templateBindingObserver.disconnect();
  }
  /** @internal */


  handleChange(source, args) {
    if (source === this.itemsBinding) {
      this.items = this.itemsBindingObserver.observe(this.source, this.originalContext);
      this.observeItems();
      this.refreshAllViews();
    } else if (source === this.templateBinding) {
      this.template = this.templateBindingObserver.observe(this.source, this.originalContext);
      this.refreshAllViews(true);
    } else {
      this.updateViews(args);
    }
  }

  observeItems(force = false) {
    if (!this.items) {
      this.items = emptyArray;
      return;
    }

    const oldObserver = this.itemsObserver;
    const newObserver = this.itemsObserver = Observable.getNotifier(this.items);
    const hasNewObserver = oldObserver !== newObserver;

    if (hasNewObserver && oldObserver !== null) {
      oldObserver.unsubscribe(this);
    }

    if (hasNewObserver || force) {
      newObserver.subscribe(this);
    }
  }

  updateViews(splices) {
    const childContext = this.childContext;
    const views = this.views;
    const totalRemoved = [];
    const bindView = this.bindView;
    let removeDelta = 0;

    for (let i = 0, ii = splices.length; i < ii; ++i) {
      const splice = splices[i];
      const removed = splice.removed;
      totalRemoved.push(...views.splice(splice.index + removeDelta, removed.length));
      removeDelta -= splice.addedCount;
    }

    const items = this.items;
    const template = this.template;

    for (let i = 0, ii = splices.length; i < ii; ++i) {
      const splice = splices[i];
      let addIndex = splice.index;
      const end = addIndex + splice.addedCount;

      for (; addIndex < end; ++addIndex) {
        const neighbor = views[addIndex];
        const location = neighbor ? neighbor.firstChild : this.location;
        const view = this.options.recycle && totalRemoved.length > 0 ? totalRemoved.shift() : template.create();
        views.splice(addIndex, 0, view);
        bindView(view, items, addIndex, childContext);
        view.insertBefore(location);
      }
    }

    for (let i = 0, ii = totalRemoved.length; i < ii; ++i) {
      totalRemoved[i].dispose();
    }

    if (this.options.positioning) {
      for (let i = 0, ii = views.length; i < ii; ++i) {
        const currentContext = views[i].context;
        currentContext.length = ii;
        currentContext.index = i;
      }
    }
  }

  refreshAllViews(templateChanged = false) {
    const items = this.items;
    const childContext = this.childContext;
    const template = this.template;
    const location = this.location;
    const bindView = this.bindView;
    let itemsLength = items.length;
    let views = this.views;
    let viewsLength = views.length;

    if (itemsLength === 0 || templateChanged) {
      // all views need to be removed
      HTMLView.disposeContiguousBatch(views);
      viewsLength = 0;
    }

    if (viewsLength === 0) {
      // all views need to be created
      this.views = views = new Array(itemsLength);

      for (let i = 0; i < itemsLength; ++i) {
        const view = template.create();
        bindView(view, items, i, childContext);
        views[i] = view;
        view.insertBefore(location);
      }
    } else {
      // attempt to reuse existing views with new data
      let i = 0;

      for (; i < itemsLength; ++i) {
        if (i < viewsLength) {
          const view = views[i];
          bindView(view, items, i, childContext);
        } else {
          const view = template.create();
          bindView(view, items, i, childContext);
          views.push(view);
          view.insertBefore(location);
        }
      }

      const removed = views.splice(i, viewsLength - i);

      for (i = 0, itemsLength = removed.length; i < itemsLength; ++i) {
        removed[i].dispose();
      }
    }
  }

  unbindAllViews() {
    const views = this.views;

    for (let i = 0, ii = views.length; i < ii; ++i) {
      views[i].unbind();
    }
  }

}
/**
 * A directive that configures list rendering.
 * @public
 */

class RepeatDirective extends HTMLDirective {
  /**
   * Creates an instance of RepeatDirective.
   * @param itemsBinding - The binding that provides the array to render.
   * @param templateBinding - The template binding used to obtain a template to render for each item in the array.
   * @param options - Options used to turn on special repeat features.
   */
  constructor(itemsBinding, templateBinding, options) {
    super();
    this.itemsBinding = itemsBinding;
    this.templateBinding = templateBinding;
    this.options = options;
    /**
     * Creates a placeholder string based on the directive's index within the template.
     * @param index - The index of the directive within the template.
     */

    this.createPlaceholder = DOM.createBlockPlaceholder;
    enableArrayObservation();
    this.isItemsBindingVolatile = Observable.isVolatileBinding(itemsBinding);
    this.isTemplateBindingVolatile = Observable.isVolatileBinding(templateBinding);
  }
  /**
   * Creates a behavior for the provided target node.
   * @param target - The node instance to create the behavior for.
   */


  createBehavior(target) {
    return new RepeatBehavior(target, this.itemsBinding, this.isItemsBindingVolatile, this.templateBinding, this.isTemplateBindingVolatile, this.options);
  }

}
/**
 * A directive that enables list rendering.
 * @param itemsBinding - The array to render.
 * @param templateOrTemplateBinding - The template or a template binding used obtain a template
 * to render for each item in the array.
 * @param options - Options used to turn on special repeat features.
 * @public
 */

function repeat(itemsBinding, templateOrTemplateBinding, options = defaultRepeatOptions) {
  const templateBinding = typeof templateOrTemplateBinding === "function" ? templateOrTemplateBinding : () => templateOrTemplateBinding;
  return new RepeatDirective(itemsBinding, templateBinding, options);
}

/**
 * Creates a function that can be used to filter a Node array, selecting only elements.
 * @param selector - An optional selector to restrict the filter to.
 * @public
 */

function elements(selector) {
  if (selector) {
    return function (value, index, array) {
      return value.nodeType === 1 && value.matches(selector);
    };
  }

  return function (value, index, array) {
    return value.nodeType === 1;
  };
}
/**
 * A base class for node observation.
 * @internal
 */

class NodeObservationBehavior {
  /**
   * Creates an instance of NodeObservationBehavior.
   * @param target - The target to assign the nodes property on.
   * @param options - The options to use in configuring node observation.
   */
  constructor(target, options) {
    this.target = target;
    this.options = options;
    this.source = null;
  }
  /**
   * Bind this behavior to the source.
   * @param source - The source to bind to.
   * @param context - The execution context that the binding is operating within.
   */


  bind(source) {
    const name = this.options.property;
    this.shouldUpdate = Observable.getAccessors(source).some(x => x.name === name);
    this.source = source;
    this.updateTarget(this.computeNodes());

    if (this.shouldUpdate) {
      this.observe();
    }
  }
  /**
   * Unbinds this behavior from the source.
   * @param source - The source to unbind from.
   */


  unbind() {
    this.updateTarget(emptyArray);
    this.source = null;

    if (this.shouldUpdate) {
      this.disconnect();
    }
  }
  /** @internal */


  handleEvent() {
    this.updateTarget(this.computeNodes());
  }

  computeNodes() {
    let nodes = this.getNodes();

    if (this.options.filter !== void 0) {
      nodes = nodes.filter(this.options.filter);
    }

    return nodes;
  }

  updateTarget(value) {
    this.source[this.options.property] = value;
  }

}

/**
 * The runtime behavior for slotted node observation.
 * @public
 */

class SlottedBehavior extends NodeObservationBehavior {
  /**
   * Creates an instance of SlottedBehavior.
   * @param target - The slot element target to observe.
   * @param options - The options to use when observing the slot.
   */
  constructor(target, options) {
    super(target, options);
  }
  /**
   * Begins observation of the nodes.
   */


  observe() {
    this.target.addEventListener("slotchange", this);
  }
  /**
   * Disconnects observation of the nodes.
   */


  disconnect() {
    this.target.removeEventListener("slotchange", this);
  }
  /**
   * Retrieves the nodes that should be assigned to the target.
   */


  getNodes() {
    return this.target.assignedNodes(this.options);
  }

}
/**
 * A directive that observes the `assignedNodes()` of a slot and updates a property
 * whenever they change.
 * @param propertyOrOptions - The options used to configure slotted node observation.
 * @public
 */

function slotted(propertyOrOptions) {
  if (typeof propertyOrOptions === "string") {
    propertyOrOptions = {
      property: propertyOrOptions
    };
  }

  return new AttachedBehaviorHTMLDirective("fast-slotted", SlottedBehavior, propertyOrOptions);
}

/**
 * The runtime behavior for child node observation.
 * @public
 */

class ChildrenBehavior extends NodeObservationBehavior {
  /**
   * Creates an instance of ChildrenBehavior.
   * @param target - The element target to observe children on.
   * @param options - The options to use when observing the element children.
   */
  constructor(target, options) {
    super(target, options);
    this.observer = null;
    options.childList = true;
  }
  /**
   * Begins observation of the nodes.
   */


  observe() {
    if (this.observer === null) {
      this.observer = new MutationObserver(this.handleEvent.bind(this));
    }

    this.observer.observe(this.target, this.options);
  }
  /**
   * Disconnects observation of the nodes.
   */


  disconnect() {
    this.observer.disconnect();
  }
  /**
   * Retrieves the nodes that should be assigned to the target.
   */


  getNodes() {
    if ("subtree" in this.options) {
      return Array.from(this.target.querySelectorAll(this.options.selector));
    }

    return Array.from(this.target.childNodes);
  }

}
/**
 * A directive that observes the `childNodes` of an element and updates a property
 * whenever they change.
 * @param propertyOrOptions - The options used to configure child node observation.
 * @public
 */

function children(propertyOrOptions) {
  if (typeof propertyOrOptions === "string") {
    propertyOrOptions = {
      property: propertyOrOptions
    };
  }

  return new AttachedBehaviorHTMLDirective("fast-children", ChildrenBehavior, propertyOrOptions);
}

/**
 * A mixin class implementing start and end elements.
 * These are generally used to decorate text elements with icons or other visual indicators.
 * @public
 */

class StartEnd {
  handleStartContentChange() {
    this.startContainer.classList.toggle("start", this.start.assignedNodes().length > 0);
  }

  handleEndContentChange() {
    this.endContainer.classList.toggle("end", this.end.assignedNodes().length > 0);
  }

}
/**
 * The template for the end element.
 * For use with {@link StartEnd}
 *
 * @public
 */

const endSlotTemplate = (context, definition) => html`<span part="end" ${ref("endContainer")} class=${x => definition.end ? "end" : void 0}><slot name="end" ${ref("end")} @slotchange="${x => x.handleEndContentChange()}">${definition.end || ""}</slot></span>`;
/**
 * The template for the start element.
 * For use with {@link StartEnd}
 *
 * @public
 */

const startSlotTemplate = (context, definition) => html`<span part="start" ${ref("startContainer")} class="${x => definition.start ? "start" : void 0}"><slot name="start" ${ref("start")} @slotchange="${x => x.handleStartContentChange()}">${definition.start || ""}</slot></span>`;
/**
 * The template for the end element.
 * For use with {@link StartEnd}
 *
 * @public
 * @deprecated - use endSlotTemplate
 */

const endTemplate = html`<span part="end" ${ref("endContainer")}><slot name="end" ${ref("end")} @slotchange="${x => x.handleEndContentChange()}"></slot></span>`;
/**
 * The template for the start element.
 * For use with {@link StartEnd}
 *
 * @public
 * @deprecated - use startSlotTemplate
 */

const startTemplate = html`<span part="start" ${ref("startContainer")}><slot name="start" ${ref("start")} @slotchange="${x => x.handleStartContentChange()}"></slot></span>`;

/**
 * The template for the {@link @microsoft/fast-foundation#(AccordionItem:class)} component.
 * @public
 */

const accordionItemTemplate = (context, definition) => html`<template class="${x => x.expanded ? "expanded" : ""}"><div class="heading" part="heading" role="heading" aria-level="${x => x.headinglevel}"><button class="button" part="button" ${ref("expandbutton")} aria-expanded="${x => x.expanded}" aria-controls="${x => x.id}-panel" id="${x => x.id}" @click="${(x, c) => x.clickHandler(c.event)}"><span class="heading-content" part="heading-content"><slot name="heading"></slot></span></button>${startSlotTemplate(context, definition)} ${endSlotTemplate(context, definition)}<span class="icon" part="icon" aria-hidden="true"><slot name="expanded-icon" part="expanded-icon">${definition.expandedIcon || ""}</slot><slot name="collapsed-icon" part="collapsed-icon">${definition.collapsedIcon || ""}</slot><span></div><div class="region" part="region" id="${x => x.id}-panel" role="region" aria-labelledby="${x => x.id}"><slot></slot></div></template>`;

/*! *****************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
function __decorate$1(decorators, target, key, desc) {
  var c = arguments.length,
      r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc,
      d;
  if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
  return c > 3 && r && Object.defineProperty(target, key, r), r;
}

/**
 * Big thanks to https://github.com/fkleuver and the https://github.com/aurelia/aurelia project
 * for the bulk of this code and many of the associated tests.
 */

const metadataByTarget = new Map();

if (!("metadata" in Reflect)) {
  Reflect.metadata = function (key, value) {
    return function (target) {
      Reflect.defineMetadata(key, value, target);
    };
  };

  Reflect.defineMetadata = function (key, value, target) {
    let metadata = metadataByTarget.get(target);

    if (metadata === void 0) {
      metadataByTarget.set(target, metadata = new Map());
    }

    metadata.set(key, value);
  };

  Reflect.getOwnMetadata = function (key, target) {
    const metadata = metadataByTarget.get(target);

    if (metadata !== void 0) {
      return metadata.get(key);
    }

    return void 0;
  };
}
/**
 * A utility class used that constructs and registers resolvers for a dependency
 * injection container. Supports a standard set of object lifetimes.
 * @public
 */


class ResolverBuilder {
  /**
   *
   * @param container - The container to create resolvers for.
   * @param key - The key to register resolvers under.
   */
  constructor(container, key) {
    this.container = container;
    this.key = key;
  }
  /**
   * Creates a resolver for an existing object instance.
   * @param value - The instance to resolve.
   * @returns The resolver.
   */


  instance(value) {
    return this.registerResolver(0
    /* instance */
    , value);
  }
  /**
   * Creates a resolver that enforces a singleton lifetime.
   * @param value - The type to create and cache the singleton for.
   * @returns The resolver.
   */


  singleton(value) {
    return this.registerResolver(1
    /* singleton */
    , value);
  }
  /**
   * Creates a resolver that creates a new instance for every dependency request.
   * @param value - The type to create instances of.
   * @returns - The resolver.
   */


  transient(value) {
    return this.registerResolver(2
    /* transient */
    , value);
  }
  /**
   * Creates a resolver that invokes a callback function for every dependency resolution
   * request, allowing custom logic to return the dependency.
   * @param value - The callback to call during resolution.
   * @returns The resolver.
   */


  callback(value) {
    return this.registerResolver(3
    /* callback */
    , value);
  }
  /**
   * Creates a resolver that invokes a callback function the first time that a dependency
   * resolution is requested. The returned value is then cached and provided for all
   * subsequent requests.
   * @param value - The callback to call during the first resolution.
   * @returns The resolver.
   */


  cachedCallback(value) {
    return this.registerResolver(3
    /* callback */
    , cacheCallbackResult(value));
  }
  /**
   * Aliases the current key to a different key.
   * @param destinationKey - The key to point the alias to.
   * @returns The resolver.
   */


  aliasTo(destinationKey) {
    return this.registerResolver(5
    /* alias */
    , destinationKey);
  }

  registerResolver(strategy, state) {
    const {
      container,
      key
    } = this;
    /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */

    this.container = this.key = void 0;
    return container.registerResolver(key, new ResolverImpl(key, strategy, state));
  }

}

function cloneArrayWithPossibleProps(source) {
  const clone = source.slice();
  const keys = Object.keys(source);
  const len = keys.length;
  let key;

  for (let i = 0; i < len; ++i) {
    key = keys[i];

    if (!isArrayIndex(key)) {
      clone[key] = source[key];
    }
  }

  return clone;
}
/**
 * A set of default resolvers useful in configuring a container.
 * @public
 */


const DefaultResolver = Object.freeze({
  /**
   * Disables auto-registration and throws for all un-registered dependencies.
   * @param key - The key to create the resolver for.
   */
  none(key) {
    throw Error(`${key.toString()} not registered, did you forget to add @singleton()?`);
  },

  /**
   * Provides default singleton resolution behavior during auto-registration.
   * @param key - The key to create the resolver for.
   * @returns The resolver.
   */
  singleton(key) {
    return new ResolverImpl(key, 1
    /* singleton */
    , key);
  },

  /**
   * Provides default transient resolution behavior during auto-registration.
   * @param key - The key to create the resolver for.
   * @returns The resolver.
   */
  transient(key) {
    return new ResolverImpl(key, 2
    /* transient */
    , key);
  }

});
/**
 * Configuration for a dependency injection container.
 * @public
 */

const ContainerConfiguration = Object.freeze({
  /**
   * The default configuration used when creating a DOM-disconnected container.
   * @remarks
   * The default creates a root container, with no parent container. It does not handle
   * owner requests and it uses singleton resolution behavior for auto-registration.
   */
  default: Object.freeze({
    parentLocator: () => null,
    responsibleForOwnerRequests: false,
    defaultResolver: DefaultResolver.singleton
  })
});
const dependencyLookup = new Map();

function getParamTypes(key) {
  return Type => {
    return Reflect.getOwnMetadata(key, Type);
  };
}

let rootDOMContainer = null;
/**
 * The gateway to dependency injection APIs.
 * @public
 */

const DI = Object.freeze({
  /**
   * Creates a new dependency injection container.
   * @param config - The configuration for the container.
   * @returns A newly created dependency injection container.
   */
  createContainer(config) {
    return new ContainerImpl(null, Object.assign({}, ContainerConfiguration.default, config));
  },

  /**
   * Finds the dependency injection container responsible for providing dependencies
   * to the specified node.
   * @param node - The node to find the responsible container for.
   * @returns The container responsible for providing dependencies to the node.
   * @remarks
   * This will be the same as the parent container if the specified node
   * does not itself host a container configured with responsibleForOwnerRequests.
   */
  findResponsibleContainer(node) {
    const owned = node.$$container$$;

    if (owned && owned.responsibleForOwnerRequests) {
      return owned;
    }

    return DI.findParentContainer(node);
  },

  /**
   * Find the dependency injection container up the DOM tree from this node.
   * @param node - The node to find the parent container for.
   * @returns The parent container of this node.
   * @remarks
   * This will be the same as the responsible container if the specified node
   * does not itself host a container configured with responsibleForOwnerRequests.
   */
  findParentContainer(node) {
    const event = new CustomEvent(DILocateParentEventType, {
      bubbles: true,
      composed: true,
      cancelable: true,
      detail: {
        container: void 0
      }
    });
    node.dispatchEvent(event);
    return event.detail.container || DI.getOrCreateDOMContainer();
  },

  /**
   * Returns a dependency injection container if one is explicitly owned by the specified
   * node. If one is not owned, then a new container is created and assigned to the node.
   * @param node - The node to find or create the container for.
   * @param config - The configuration for the container if one needs to be created.
   * @returns The located or created container.
   * @remarks
   * This API does not search for a responsible or parent container. It looks only for a container
   * directly defined on the specified node and creates one at that location if one does not
   * already exist.
   */
  getOrCreateDOMContainer(node, config) {
    if (!node) {
      return rootDOMContainer || (rootDOMContainer = new ContainerImpl(null, Object.assign({}, ContainerConfiguration.default, config, {
        parentLocator: () => null
      })));
    }

    return node.$$container$$ || new ContainerImpl(node, Object.assign({}, ContainerConfiguration.default, config, {
      parentLocator: DI.findParentContainer
    }));
  },

  /**
   * Gets the "design:paramtypes" metadata for the specified type.
   * @param Type - The type to get the metadata for.
   * @returns The metadata array or undefined if no metadata is found.
   */
  getDesignParamtypes: getParamTypes("design:paramtypes"),

  /**
   * Gets the "di:paramtypes" metadata for the specified type.
   * @param Type - The type to get the metadata for.
   * @returns The metadata array or undefined if no metadata is found.
   */
  getAnnotationParamtypes: getParamTypes("di:paramtypes"),

  /**
   *
   * @param Type - Gets the "di:paramtypes" metadata for the specified type. If none is found,
   * an empty metadata array is created and added.
   * @returns The metadata array.
   */
  getOrCreateAnnotationParamTypes(Type) {
    let annotationParamtypes = this.getAnnotationParamtypes(Type);

    if (annotationParamtypes === void 0) {
      Reflect.defineMetadata("di:paramtypes", annotationParamtypes = [], Type);
    }

    return annotationParamtypes;
  },

  /**
   * Gets the dependency keys representing what is needed to instantiate the specified type.
   * @param Type - The type to get the dependencies for.
   * @returns An array of dependency keys.
   */
  getDependencies(Type) {
    // Note: Every detail of this getDependencies method is pretty deliberate at the moment, and probably not yet 100% tested from every possible angle,
    // so be careful with making changes here as it can have a huge impact on complex end user apps.
    // Preferably, only make changes to the dependency resolution process via a RFC.
    let dependencies = dependencyLookup.get(Type);

    if (dependencies === void 0) {
      // Type.length is the number of constructor parameters. If this is 0, it could mean the class has an empty constructor
      // but it could also mean the class has no constructor at all (in which case it inherits the constructor from the prototype).
      // Non-zero constructor length + no paramtypes means emitDecoratorMetadata is off, or the class has no decorator.
      // We're not doing anything with the above right now, but it's good to keep in mind for any future issues.
      const inject = Type.inject;

      if (inject === void 0) {
        // design:paramtypes is set by tsc when emitDecoratorMetadata is enabled.
        const designParamtypes = DI.getDesignParamtypes(Type); // di:paramtypes is set by the parameter decorator from DI.createInterface or by @inject

        const annotationParamtypes = DI.getAnnotationParamtypes(Type);

        if (designParamtypes === void 0) {
          if (annotationParamtypes === void 0) {
            // Only go up the prototype if neither static inject nor any of the paramtypes is defined, as
            // there is no sound way to merge a type's deps with its prototype's deps
            const Proto = Object.getPrototypeOf(Type);

            if (typeof Proto === "function" && Proto !== Function.prototype) {
              dependencies = cloneArrayWithPossibleProps(DI.getDependencies(Proto));
            } else {
              dependencies = [];
            }
          } else {
            // No design:paramtypes so just use the di:paramtypes
            dependencies = cloneArrayWithPossibleProps(annotationParamtypes);
          }
        } else if (annotationParamtypes === void 0) {
          // No di:paramtypes so just use the design:paramtypes
          dependencies = cloneArrayWithPossibleProps(designParamtypes);
        } else {
          // We've got both, so merge them (in case of conflict on same index, di:paramtypes take precedence)
          dependencies = cloneArrayWithPossibleProps(designParamtypes);
          let len = annotationParamtypes.length;
          let auAnnotationParamtype;

          for (let i = 0; i < len; ++i) {
            auAnnotationParamtype = annotationParamtypes[i];

            if (auAnnotationParamtype !== void 0) {
              dependencies[i] = auAnnotationParamtype;
            }
          }

          const keys = Object.keys(annotationParamtypes);
          len = keys.length;
          let key;

          for (let i = 0; i < len; ++i) {
            key = keys[i];

            if (!isArrayIndex(key)) {
              dependencies[key] = annotationParamtypes[key];
            }
          }
        }
      } else {
        // Ignore paramtypes if we have static inject
        dependencies = cloneArrayWithPossibleProps(inject);
      }

      dependencyLookup.set(Type, dependencies);
    }

    return dependencies;
  },

  /**
   * Defines a property on a web component class. The value of this property will
   * be resolved from the dependency injection container responsible for the element
   * instance, based on where it is connected in the DOM.
   * @param target - The target to define the property on.
   * @param propertyName - The name of the property to define.
   * @param key - The dependency injection key.
   * @param respectConnection - Indicates whether or not to update the property value if the
   * hosting component is disconnected and then re-connected at a different location in the DOM.
   * @remarks
   * The respectConnection option is only applicable to elements that descend from FASTElement.
   */
  defineProperty(target, propertyName, key, respectConnection = false) {
    const diPropertyKey = `$di_${propertyName}`;
    Reflect.defineProperty(target, propertyName, {
      get: function () {
        let value = this[diPropertyKey];

        if (value === void 0) {
          const container = this instanceof HTMLElement ? DI.findResponsibleContainer(this) : DI.getOrCreateDOMContainer();
          value = container.get(key);
          this[diPropertyKey] = value;

          if (respectConnection && this instanceof FASTElement) {
            const notifier = this.$fastController;

            const handleChange = () => {
              const newContainer = DI.findResponsibleContainer(this);
              const newValue = newContainer.get(key);
              const oldValue = this[diPropertyKey];

              if (newValue !== oldValue) {
                this[diPropertyKey] = value;
                notifier.notify(propertyName);
              }
            };

            notifier.subscribe({
              handleChange
            }, "isConnected");
          }
        }

        return value;
      }
    });
  },

  /**
   * Creates a dependency injection key.
   * @param nameConfigOrCallback - A friendly name for the key or a lambda that configures a
   * default resolution for the dependency.
   * @param configuror - If a friendly name was provided for the first parameter, then an optional
   * lambda that configures a default resolution for the dependency can be provided second.
   * @returns The created key.
   * @remarks
   * The created key can be used as a property decorator or constructor parameter decorator,
   * in addition to its standard use in an inject array or through direct container APIs.
   */
  createInterface(nameConfigOrCallback, configuror) {
    const configure = typeof nameConfigOrCallback === "function" ? nameConfigOrCallback : configuror;
    const friendlyName = typeof nameConfigOrCallback === "string" ? nameConfigOrCallback : nameConfigOrCallback && "friendlyName" in nameConfigOrCallback ? nameConfigOrCallback.friendlyName || defaultFriendlyName : defaultFriendlyName;
    const respectConnection = typeof nameConfigOrCallback === "string" ? false : nameConfigOrCallback && "respectConnection" in nameConfigOrCallback ? nameConfigOrCallback.respectConnection || false : false;

    const Interface = function (target, property, index) {
      if (target == null || new.target !== undefined) {
        throw new Error(`No registration for interface: '${Interface.friendlyName}'`);
      }

      if (property) {
        DI.defineProperty(target, property, Interface, respectConnection);
      } else {
        const annotationParamtypes = DI.getOrCreateAnnotationParamTypes(target);
        annotationParamtypes[index] = Interface;
      }
    };

    Interface.$isInterface = true;
    Interface.friendlyName = friendlyName == null ? "(anonymous)" : friendlyName;

    if (configure != null) {
      Interface.register = function (container, key) {
        return configure(new ResolverBuilder(container, key !== null && key !== void 0 ? key : Interface));
      };
    }

    Interface.toString = function toString() {
      return `InterfaceSymbol<${Interface.friendlyName}>`;
    };

    return Interface;
  },

  /**
   * A decorator that specifies what to inject into its target.
   * @param dependencies - The dependencies to inject.
   * @returns The decorator to be applied to the target class.
   * @remarks
   * The decorator can be used to decorate a class, listing all of the classes dependencies.
   * Or it can be used to decorate a constructor paramter, indicating what to inject for that
   * parameter.
   * Or it can be used for a web component property, indicating what that property should resolve to.
   */
  inject(...dependencies) {
    return function (target, key, descriptor) {
      if (typeof descriptor === "number") {
        // It's a parameter decorator.
        const annotationParamtypes = DI.getOrCreateAnnotationParamTypes(target);
        const dep = dependencies[0];

        if (dep !== void 0) {
          annotationParamtypes[descriptor] = dep;
        }
      } else if (key) {
        DI.defineProperty(target, key, dependencies[0]);
      } else {
        const annotationParamtypes = descriptor ? DI.getOrCreateAnnotationParamTypes(descriptor.value) : DI.getOrCreateAnnotationParamTypes(target);
        let dep;

        for (let i = 0; i < dependencies.length; ++i) {
          dep = dependencies[i];

          if (dep !== void 0) {
            annotationParamtypes[i] = dep;
          }
        }
      }
    };
  },

  /**
   * Registers the `target` class as a transient dependency; each time the dependency is resolved
   * a new instance will be created.
   *
   * @param target - The class / constructor function to register as transient.
   * @returns The same class, with a static `register` method that takes a container and returns the appropriate resolver.
   *
   * @example
   * On an existing class
   * ```ts
   * class Foo { }
   * DI.transient(Foo);
   * ```
   *
   * @example
   * Inline declaration
   *
   * ```ts
   * const Foo = DI.transient(class { });
   * // Foo is now strongly typed with register
   * Foo.register(container);
   * ```
   *
   * @public
   */
  transient(target) {
    target.register = function register(container) {
      const registration = Registration.transient(target, target);
      return registration.register(container);
    };

    target.registerInRequestor = false;
    return target;
  },

  /**
   * Registers the `target` class as a singleton dependency; the class will only be created once. Each
   * consecutive time the dependency is resolved, the same instance will be returned.
   *
   * @param target - The class / constructor function to register as a singleton.
   * @returns The same class, with a static `register` method that takes a container and returns the appropriate resolver.
   * @example
   * On an existing class
   * ```ts
   * class Foo { }
   * DI.singleton(Foo);
   * ```
   *
   * @example
   * Inline declaration
   * ```ts
   * const Foo = DI.singleton(class { });
   * // Foo is now strongly typed with register
   * Foo.register(container);
   * ```
   *
   * @public
   */
  singleton(target, options = defaultSingletonOptions) {
    target.register = function register(container) {
      const registration = Registration.singleton(target, target);
      return registration.register(container);
    };

    target.registerInRequestor = options.scoped;
    return target;
  }

});
/**
 * The interface key that resolves the dependency injection container itself.
 * @public
 */

const Container = DI.createInterface("Container");
/**
 * A decorator that specifies what to inject into its target.
 * @param dependencies - The dependencies to inject.
 * @returns The decorator to be applied to the target class.
 * @remarks
 * The decorator can be used to decorate a class, listing all of the classes dependencies.
 * Or it can be used to decorate a constructor paramter, indicating what to inject for that
 * parameter.
 * Or it can be used for a web component property, indicating what that property should resolve to.
 *
 * @public
 */


DI.inject;
const defaultSingletonOptions = {
  scoped: false
};
/** @internal */


class ResolverImpl {
  constructor(key, strategy, state) {
    this.key = key;
    this.strategy = strategy;
    this.state = state;
    this.resolving = false;
  }

  get $isResolver() {
    return true;
  }

  register(container) {
    return container.registerResolver(this.key, this);
  }

  resolve(handler, requestor) {
    switch (this.strategy) {
      case 0
      /* instance */
      :
        return this.state;

      case 1
      /* singleton */
      :
        {
          if (this.resolving) {
            throw new Error(`Cyclic dependency found: ${this.state.name}`);
          }

          this.resolving = true;
          this.state = handler.getFactory(this.state).construct(requestor);
          this.strategy = 0
          /* instance */
          ;
          this.resolving = false;
          return this.state;
        }

      case 2
      /* transient */
      :
        {
          // Always create transients from the requesting container
          const factory = handler.getFactory(this.state);

          if (factory === null) {
            throw new Error(`Resolver for ${String(this.key)} returned a null factory`);
          }

          return factory.construct(requestor);
        }

      case 3
      /* callback */
      :
        return this.state(handler, requestor, this);

      case 4
      /* array */
      :
        return this.state[0].resolve(handler, requestor);

      case 5
      /* alias */
      :
        return requestor.get(this.state);

      default:
        throw new Error(`Invalid resolver strategy specified: ${this.strategy}.`);
    }
  }

  getFactory(container) {
    var _a, _b, _c;

    switch (this.strategy) {
      case 1
      /* singleton */
      :
      case 2
      /* transient */
      :
        return container.getFactory(this.state);

      case 5
      /* alias */
      :
        return (_c = (_b = (_a = container.getResolver(this.state)) === null || _a === void 0 ? void 0 : _a.getFactory) === null || _b === void 0 ? void 0 : _b.call(_a, container)) !== null && _c !== void 0 ? _c : null;

      default:
        return null;
    }
  }

}

function containerGetKey(d) {
  return this.get(d);
}

function transformInstance(inst, transform) {
  return transform(inst);
}
/** @internal */


class FactoryImpl {
  constructor(Type, dependencies) {
    this.Type = Type;
    this.dependencies = dependencies;
    this.transformers = null;
  }

  construct(container, dynamicDependencies) {
    let instance;

    if (dynamicDependencies === void 0) {
      instance = new this.Type(...this.dependencies.map(containerGetKey, container));
    } else {
      instance = new this.Type(...this.dependencies.map(containerGetKey, container), ...dynamicDependencies);
    }

    if (this.transformers == null) {
      return instance;
    }

    return this.transformers.reduce(transformInstance, instance);
  }

  registerTransformer(transformer) {
    (this.transformers || (this.transformers = [])).push(transformer);
  }

}
const containerResolver = {
  $isResolver: true,

  resolve(handler, requestor) {
    return requestor;
  }

};

function isRegistry(obj) {
  return typeof obj.register === "function";
}

function isSelfRegistry(obj) {
  return isRegistry(obj) && typeof obj.registerInRequestor === "boolean";
}

function isRegisterInRequester(obj) {
  return isSelfRegistry(obj) && obj.registerInRequestor;
}

function isClass(obj) {
  return obj.prototype !== void 0;
}

const InstrinsicTypeNames = new Set(["Array", "ArrayBuffer", "Boolean", "DataView", "Date", "Error", "EvalError", "Float32Array", "Float64Array", "Function", "Int8Array", "Int16Array", "Int32Array", "Map", "Number", "Object", "Promise", "RangeError", "ReferenceError", "RegExp", "Set", "SharedArrayBuffer", "String", "SyntaxError", "TypeError", "Uint8Array", "Uint8ClampedArray", "Uint16Array", "Uint32Array", "URIError", "WeakMap", "WeakSet"]);
const DILocateParentEventType = "__DI_LOCATE_PARENT__";
const factories = new Map();
/**
 * @internal
 */

class ContainerImpl {
  constructor(owner, config) {
    this.owner = owner;
    this.config = config;
    this._parent = void 0;
    this.registerDepth = 0;
    this.context = null;

    if (owner !== null) {
      owner.$$container$$ = this;
    }

    this.resolvers = new Map();
    this.resolvers.set(Container, containerResolver);

    if (owner instanceof Node) {
      owner.addEventListener(DILocateParentEventType, e => {
        if (e.composedPath()[0] !== this.owner) {
          e.detail.container = this;
          e.stopImmediatePropagation();
        }
      });
    }
  }

  get parent() {
    if (this._parent === void 0) {
      this._parent = this.config.parentLocator(this.owner);
    }

    return this._parent;
  }

  get depth() {
    return this.parent === null ? 0 : this.parent.depth + 1;
  }

  get responsibleForOwnerRequests() {
    return this.config.responsibleForOwnerRequests;
  }

  registerWithContext(context, ...params) {
    this.context = context;
    this.register(...params);
    this.context = null;
    return this;
  }

  register(...params) {
    if (++this.registerDepth === 100) {
      throw new Error("Unable to autoregister dependency"); // Most likely cause is trying to register a plain object that does not have a
      // register method and is not a class constructor
    }

    let current;
    let keys;
    let value;
    let j;
    let jj;
    const context = this.context;

    for (let i = 0, ii = params.length; i < ii; ++i) {
      current = params[i];

      if (!isObject(current)) {
        continue;
      }

      if (isRegistry(current)) {
        current.register(this, context);
      } else if (isClass(current)) {
        Registration.singleton(current, current).register(this);
      } else {
        keys = Object.keys(current);
        j = 0;
        jj = keys.length;

        for (; j < jj; ++j) {
          value = current[keys[j]];

          if (!isObject(value)) {
            continue;
          } // note: we could remove this if-branch and call this.register directly
          // - the extra check is just a perf tweak to create fewer unnecessary arrays by the spread operator


          if (isRegistry(value)) {
            value.register(this, context);
          } else {
            this.register(value);
          }
        }
      }
    }

    --this.registerDepth;
    return this;
  }

  registerResolver(key, resolver) {
    validateKey(key);
    const resolvers = this.resolvers;
    const result = resolvers.get(key);

    if (result == null) {
      resolvers.set(key, resolver);
    } else if (result instanceof ResolverImpl && result.strategy === 4
    /* array */
    ) {
      result.state.push(resolver);
    } else {
      resolvers.set(key, new ResolverImpl(key, 4
      /* array */
      , [result, resolver]));
    }

    return resolver;
  }

  registerTransformer(key, transformer) {
    const resolver = this.getResolver(key);

    if (resolver == null) {
      return false;
    }

    if (resolver.getFactory) {
      const factory = resolver.getFactory(this);

      if (factory == null) {
        return false;
      } // This type cast is a bit of a hacky one, necessary due to the duplicity of IResolverLike.
      // Problem is that that interface's type arg can be of type Key, but the getFactory method only works on
      // type Constructable. So the return type of that optional method has this additional constraint, which
      // seems to confuse the type checker.


      factory.registerTransformer(transformer);
      return true;
    }

    return false;
  }

  getResolver(key, autoRegister = true) {
    validateKey(key);

    if (key.resolve !== void 0) {
      return key;
    }
    /* eslint-disable-next-line @typescript-eslint/no-this-alias */


    let current = this;
    let resolver;

    while (current != null) {
      resolver = current.resolvers.get(key);

      if (resolver == null) {
        if (current.parent == null) {
          const handler = isRegisterInRequester(key) ? this : current;
          return autoRegister ? this.jitRegister(key, handler) : null;
        }

        current = current.parent;
      } else {
        return resolver;
      }
    }

    return null;
  }

  has(key, searchAncestors = false) {
    return this.resolvers.has(key) ? true : searchAncestors && this.parent != null ? this.parent.has(key, true) : false;
  }

  get(key) {
    validateKey(key);

    if (key.$isResolver) {
      return key.resolve(this, this);
    }
    /* eslint-disable-next-line @typescript-eslint/no-this-alias */


    let current = this;
    let resolver;

    while (current != null) {
      resolver = current.resolvers.get(key);

      if (resolver == null) {
        if (current.parent == null) {
          const handler = isRegisterInRequester(key) ? this : current;
          resolver = this.jitRegister(key, handler);
          return resolver.resolve(current, this);
        }

        current = current.parent;
      } else {
        return resolver.resolve(current, this);
      }
    }

    throw new Error(`Unable to resolve key: ${key}`);
  }

  getAll(key, searchAncestors = false) {
    validateKey(key);
    /* eslint-disable-next-line @typescript-eslint/no-this-alias */

    const requestor = this;
    let current = requestor;
    let resolver;

    if (searchAncestors) {
      let resolutions = emptyArray;

      while (current != null) {
        resolver = current.resolvers.get(key);

        if (resolver != null) {
          resolutions = resolutions.concat(
          /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */
          buildAllResponse(resolver, current, requestor));
        }

        current = current.parent;
      }

      return resolutions;
    } else {
      while (current != null) {
        resolver = current.resolvers.get(key);

        if (resolver == null) {
          current = current.parent;

          if (current == null) {
            return emptyArray;
          }
        } else {
          return buildAllResponse(resolver, current, requestor);
        }
      }
    }

    return emptyArray;
  }

  getFactory(Type) {
    let factory = factories.get(Type);

    if (factory === void 0) {
      if (isNativeFunction(Type)) {
        throw new Error(`${Type.name} is a native function and therefore cannot be safely constructed by DI. If this is intentional, please use a callback or cachedCallback resolver.`);
      }

      factories.set(Type, factory = new FactoryImpl(Type, DI.getDependencies(Type)));
    }

    return factory;
  }

  registerFactory(key, factory) {
    factories.set(key, factory);
  }

  createChild(config) {
    return new ContainerImpl(null, Object.assign({}, this.config, config, {
      parentLocator: () => this
    }));
  }

  jitRegister(keyAsValue, handler) {
    if (typeof keyAsValue !== "function") {
      throw new Error(`Attempted to jitRegister something that is not a constructor: '${keyAsValue}'. Did you forget to register this dependency?`);
    }

    if (InstrinsicTypeNames.has(keyAsValue.name)) {
      throw new Error(`Attempted to jitRegister an intrinsic type: ${keyAsValue.name}. Did you forget to add @inject(Key)`);
    }

    if (isRegistry(keyAsValue)) {
      const registrationResolver = keyAsValue.register(handler);

      if (!(registrationResolver instanceof Object) || registrationResolver.resolve == null) {
        const newResolver = handler.resolvers.get(keyAsValue);

        if (newResolver != void 0) {
          return newResolver;
        }

        throw new Error("A valid resolver was not returned from the static register method");
      }

      return registrationResolver;
    } else if (keyAsValue.$isInterface) {
      throw new Error(`Attempted to jitRegister an interface: ${keyAsValue.friendlyName}`);
    } else {
      const resolver = this.config.defaultResolver(keyAsValue, handler);
      handler.resolvers.set(keyAsValue, resolver);
      return resolver;
    }
  }

}
const cache = new WeakMap();

function cacheCallbackResult(fun) {
  return function (handler, requestor, resolver) {
    if (cache.has(resolver)) {
      return cache.get(resolver);
    }

    const t = fun(handler, requestor, resolver);
    cache.set(resolver, t);
    return t;
  };
}
/**
 * You can use the resulting Registration of any of the factory methods
 * to register with the container.
 *
 * @example
 * ```
 * class Foo {}
 * const container = DI.createContainer();
 * container.register(Registration.instance(Foo, new Foo()));
 * container.get(Foo);
 * ```
 *
 * @public
 */


const Registration = Object.freeze({
  /**
   * Allows you to pass an instance.
   * Every time you request this {@link Key} you will get this instance back.
   *
   * @example
   * ```
   * Registration.instance(Foo, new Foo()));
   * ```
   *
   * @param key - The key to register the instance under.
   * @param value - The instance to return when the key is requested.
   */
  instance(key, value) {
    return new ResolverImpl(key, 0
    /* instance */
    , value);
  },

  /**
   * Creates an instance from the class.
   * Every time you request this {@link Key} you will get the same one back.
   *
   * @example
   * ```
   * Registration.singleton(Foo, Foo);
   * ```
   *
   * @param key - The key to register the singleton under.
   * @param value - The class to instantiate as a singleton when first requested.
   */
  singleton(key, value) {
    return new ResolverImpl(key, 1
    /* singleton */
    , value);
  },

  /**
   * Creates an instance from a class.
   * Every time you request this {@link Key} you will get a new instance.
   *
   * @example
   * ```
   * Registration.instance(Foo, Foo);
   * ```
   *
   * @param key - The key to register the instance type under.
   * @param value - The class to instantiate each time the key is requested.
   */
  transient(key, value) {
    return new ResolverImpl(key, 2
    /* transient */
    , value);
  },

  /**
   * Delegates to a callback function to provide the dependency.
   * Every time you request this {@link Key} the callback will be invoked to provide
   * the dependency.
   *
   * @example
   * ```
   * Registration.callback(Foo, () => new Foo());
   * Registration.callback(Bar, (c: Container) => new Bar(c.get(Foo)));
   * ```
   *
   * @param key - The key to register the callback for.
   * @param callback - The function that is expected to return the dependency.
   */
  callback(key, callback) {
    return new ResolverImpl(key, 3
    /* callback */
    , callback);
  },

  /**
   * Delegates to a callback function to provide the dependency and then caches the
   * dependency for future requests.
   *
   * @example
   * ```
   * Registration.cachedCallback(Foo, () => new Foo());
   * Registration.cachedCallback(Bar, (c: Container) => new Bar(c.get(Foo)));
   * ```
   *
   * @param key - The key to register the callback for.
   * @param callback - The function that is expected to return the dependency.
   * @remarks
   * If you pass the same Registration to another container, the same cached value will be used.
   * Should all references to the resolver returned be removed, the cache will expire.
   */
  cachedCallback(key, callback) {
    return new ResolverImpl(key, 3
    /* callback */
    , cacheCallbackResult(callback));
  },

  /**
   * Creates an alternate {@link Key} to retrieve an instance by.
   *
   * @example
   * ```
   * Register.singleton(Foo, Foo)
   * Register.aliasTo(Foo, MyFoos);
   *
   * container.getAll(MyFoos) // contains an instance of Foo
   * ```
   *
   * @param originalKey - The original key that has been registered.
   * @param aliasKey - The alias to the original key.
   */
  aliasTo(originalKey, aliasKey) {
    return new ResolverImpl(aliasKey, 5
    /* alias */
    , originalKey);
  }

});
/** @internal */

function validateKey(key) {
  if (key === null || key === void 0) {
    throw new Error("key/value cannot be null or undefined. Are you trying to inject/register something that doesn't exist with DI?");
  }
}

function buildAllResponse(resolver, handler, requestor) {
  if (resolver instanceof ResolverImpl && resolver.strategy === 4
  /* array */
  ) {
    const state = resolver.state;
    let i = state.length;
    const results = new Array(i);

    while (i--) {
      results[i] = state[i].resolve(handler, requestor);
    }

    return results;
  }

  return [resolver.resolve(handler, requestor)];
}

const defaultFriendlyName = "(anonymous)";

function isObject(value) {
  return typeof value === "object" && value !== null || typeof value === "function";
}
/**
 * Determine whether the value is a native function.
 *
 * @param fn - The function to check.
 * @returns `true` is the function is a native function, otherwise `false`
 */


const isNativeFunction = function () {
  const lookup = new WeakMap();
  let isNative = false;
  let sourceText = "";
  let i = 0;
  return function (fn) {
    isNative = lookup.get(fn);

    if (isNative === void 0) {
      sourceText = fn.toString();
      i = sourceText.length; // http://www.ecma-international.org/ecma-262/#prod-NativeFunction

      isNative = // 29 is the length of 'function () { [native code] }' which is the smallest length of a native function string
      i >= 29 && // 100 seems to be a safe upper bound of the max length of a native function. In Chrome and FF it's 56, in Edge it's 61.
      i <= 100 && // This whole heuristic *could* be tricked by a comment. Do we need to care about that?
      sourceText.charCodeAt(i - 1) === 0x7d && // }
      // TODO: the spec is a little vague about the precise constraints, so we do need to test this across various browsers to make sure just one whitespace is a safe assumption.
      sourceText.charCodeAt(i - 2) <= 0x20 && // whitespace
      sourceText.charCodeAt(i - 3) === 0x5d && // ]
      sourceText.charCodeAt(i - 4) === 0x65 && // e
      sourceText.charCodeAt(i - 5) === 0x64 && // d
      sourceText.charCodeAt(i - 6) === 0x6f && // o
      sourceText.charCodeAt(i - 7) === 0x63 && // c
      sourceText.charCodeAt(i - 8) === 0x20 && //
      sourceText.charCodeAt(i - 9) === 0x65 && // e
      sourceText.charCodeAt(i - 10) === 0x76 && // v
      sourceText.charCodeAt(i - 11) === 0x69 && // i
      sourceText.charCodeAt(i - 12) === 0x74 && // t
      sourceText.charCodeAt(i - 13) === 0x61 && // a
      sourceText.charCodeAt(i - 14) === 0x6e && // n
      sourceText.charCodeAt(i - 15) === 0x58; // [

      lookup.set(fn, isNative);
    }

    return isNative;
  };
}();

const isNumericLookup = {};

function isArrayIndex(value) {
  switch (typeof value) {
    case "number":
      return value >= 0 && (value | 0) === value;

    case "string":
      {
        const result = isNumericLookup[value];

        if (result !== void 0) {
          return result;
        }

        const length = value.length;

        if (length === 0) {
          return isNumericLookup[value] = false;
        }

        let ch = 0;

        for (let i = 0; i < length; ++i) {
          ch = value.charCodeAt(i);

          if (i === 0 && ch === 0x30 && length > 1
          /* must not start with 0 */
          || ch < 0x30
          /* 0 */
          || ch > 0x39
          /* 9 */
          ) {
            return isNumericLookup[value] = false;
          }
        }

        return isNumericLookup[value] = true;
      }

    default:
      return false;
  }
}

function presentationKeyFromTag(tagName) {
  return `${tagName.toLowerCase()}:presentation`;
}

const presentationRegistry = new Map();
/**
 * An API gateway to component presentation features.
 * @public
 */

const ComponentPresentation = Object.freeze({
  /**
   * Defines a component presentation for an element.
   * @param tagName - The element name to define the presentation for.
   * @param presentation - The presentation that will be applied to matching elements.
   * @param container - The dependency injection container to register the configuration in.
   * @public
   */
  define(tagName, presentation, container) {
    const key = presentationKeyFromTag(tagName);
    const existing = presentationRegistry.get(key);

    if (existing === void 0) {
      presentationRegistry.set(key, presentation);
    } else {
      // false indicates that we have more than one presentation
      // registered for a tagName and we must resolve through DI
      presentationRegistry.set(key, false);
    }

    container.register(Registration.instance(key, presentation));
  },

  /**
   * Finds a component presentation for the specified element name,
   * searching the DOM hierarchy starting from the provided element.
   * @param tagName - The name of the element to locate the presentation for.
   * @param element - The element to begin the search from.
   * @returns The component presentation or null if none is found.
   * @public
   */
  forTag(tagName, element) {
    const key = presentationKeyFromTag(tagName);
    const existing = presentationRegistry.get(key);

    if (existing === false) {
      const container = DI.findResponsibleContainer(element);
      return container.get(key);
    }

    return existing || null;
  }

});
/**
 * The default implementation of ComponentPresentation, used by FoundationElement.
 * @public
 */

class DefaultComponentPresentation {
  /**
   * Creates an instance of DefaultComponentPresentation.
   * @param template - The template to apply to the element.
   * @param styles - The styles to apply to the element.
   * @public
   */
  constructor(template, styles) {
    this.template = template || null;
    this.styles = styles === void 0 ? null : Array.isArray(styles) ? ElementStyles.create(styles) : styles instanceof ElementStyles ? styles : ElementStyles.create([styles]);
  }
  /**
   * Applies the presentation details to the specified element.
   * @param element - The element to apply the presentation details to.
   * @public
   */


  applyTo(element) {
    const controller = element.$fastController;

    if (controller.template === null) {
      controller.template = this.template;
    }

    if (controller.styles === null) {
      controller.styles = this.styles;
    }
  }

}

/**
 * Defines a foundation element class that:
 * 1. Connects the element to its ComponentPresentation
 * 2. Allows resolving the element template from the instance or ComponentPresentation
 * 3. Allows resolving the element styles from the instance or ComponentPresentation
 *
 * @public
 */

class FoundationElement extends FASTElement {
  constructor() {
    super(...arguments);
    this._presentation = void 0;
  }
  /**
   * A property which resolves the ComponentPresentation instance
   * for the current component.
   * @public
   */


  get $presentation() {
    if (this._presentation === void 0) {
      this._presentation = ComponentPresentation.forTag(this.tagName, this);
    }

    return this._presentation;
  }

  templateChanged() {
    if (this.template !== undefined) {
      this.$fastController.template = this.template;
    }
  }

  stylesChanged() {
    if (this.styles !== undefined) {
      this.$fastController.styles = this.styles;
    }
  }
  /**
   * The connected callback for this FASTElement.
   * @remarks
   * This method is invoked by the platform whenever this FoundationElement
   * becomes connected to the document.
   * @public
   */


  connectedCallback() {
    if (this.$presentation !== null) {
      this.$presentation.applyTo(this);
    }

    super.connectedCallback();
  }
  /**
   * Defines an element registry function with a set of element definition defaults.
   * @param elementDefinition - The definition of the element to create the registry
   * function for.
   * @public
   */


  static compose(elementDefinition) {
    return (overrideDefinition = {}) => new FoundationElementRegistry(this === FoundationElement ? class extends FoundationElement {} : this, elementDefinition, overrideDefinition);
  }

}

__decorate$1([observable], FoundationElement.prototype, "template", void 0);

__decorate$1([observable], FoundationElement.prototype, "styles", void 0);

function resolveOption(option, context, definition) {
  if (typeof option === "function") {
    return option(context, definition);
  }

  return option;
}
/**
 * Registry capable of defining presentation properties for a DOM Container hierarchy.
 *
 * @internal
 */

/* eslint-disable @typescript-eslint/no-unused-vars */


class FoundationElementRegistry {
  constructor(type, elementDefinition, overrideDefinition) {
    this.type = type;
    this.elementDefinition = elementDefinition;
    this.overrideDefinition = overrideDefinition;
    this.definition = Object.assign(Object.assign({}, this.elementDefinition), this.overrideDefinition);
  }

  register(container, context) {
    const definition = this.definition;
    const overrideDefinition = this.overrideDefinition;
    const prefix = definition.prefix || context.elementPrefix;
    const name = `${prefix}-${definition.baseName}`;
    context.tryDefineElement({
      name,
      type: this.type,
      baseClass: this.elementDefinition.baseClass,
      callback: x => {
        const presentation = new DefaultComponentPresentation(resolveOption(definition.template, x, definition), resolveOption(definition.styles, x, definition));
        x.definePresentation(presentation);
        let shadowOptions = resolveOption(definition.shadowOptions, x, definition);

        if (x.shadowRootMode) {
          // If the design system has overridden the shadow root mode, we need special handling.
          if (shadowOptions) {
            // If there are shadow options present in the definition, then
            // either the component itself has specified an option or the
            // registry function has overridden it.
            if (!overrideDefinition.shadowOptions) {
              // There were shadow options provided by the component and not overridden by
              // the registry.
              shadowOptions.mode = x.shadowRootMode;
            }
          } else if (shadowOptions !== null) {
            // If the component author did not provide shadow options,
            // and did not null them out (light dom opt-in) then they
            // were relying on the FASTElement default. So, if the
            // design system provides a mode, we need to create the options
            // to override the default.
            shadowOptions = {
              mode: x.shadowRootMode
            };
          }
        }

        x.defineElement({
          elementOptions: resolveOption(definition.elementOptions, x, definition),
          shadowOptions,
          attributes: resolveOption(definition.attributes, x, definition)
        });
      }
    });
  }

}
/* eslint-enable @typescript-eslint/no-unused-vars */

/**
 * Apply mixins to a constructor.
 * Sourced from {@link https://www.typescriptlang.org/docs/handbook/mixins.html | TypeScript Documentation }.
 * @public
 */
function applyMixins(derivedCtor, ...baseCtors) {
  baseCtors.forEach(baseCtor => {
    Object.getOwnPropertyNames(baseCtor.prototype).forEach(name => {
      if (name !== "constructor") {
        Object.defineProperty(derivedCtor.prototype, name,
        /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */
        Object.getOwnPropertyDescriptor(baseCtor.prototype, name));
      }
    });

    if (baseCtor.attributes) {
      const existing = derivedCtor.attributes || [];
      derivedCtor.attributes = existing.concat(baseCtor.attributes);
    }
  });
}

/**
 * An individual item in an {@link @microsoft/fast-foundation#(Accordion:class) }.
 *
 * @slot start - Content which can be provided between the heading and the icon
 * @slot end - Content which can be provided between the start slot and icon
 * @slot heading - Content which serves as the accordion item heading and text of the expand button
 * @slot - The default slot for accordion item content
 * @slot expanded-icon - The expanded icon
 * @slot collapsed-icon - The collapsed icon
 * @fires change - Fires a custom 'change' event when the button is invoked
 * @csspart heading - Wraps the button
 * @csspart button - The button which serves to invoke the item
 * @csspart heading-content - Wraps the slot for the heading content within the button
 * @csspart icon - The icon container
 * @csspart expanded-icon - The expanded icon slot
 * @csspart collapsed-icon - The collapsed icon
 * @csspart region - The wrapper for the accordion item content
 *
 * @public
 */

class AccordionItem extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * Configures the {@link https://www.w3.org/TR/wai-aria-1.1/#aria-level | level} of the
     * heading element.
     *
     * @defaultValue 2
     * @public
     * @remarks
     * HTML attribute: heading-level
     */

    this.headinglevel = 2;
    /**
     * Expands or collapses the item.
     *
     * @public
     * @remarks
     * HTML attribute: expanded
     */

    this.expanded = false;
    /**
     * @internal
     */

    this.clickHandler = e => {
      this.expanded = !this.expanded;
      this.change();
    };

    this.change = () => {
      this.$emit("change");
    };
  }

}

__decorate$1([attr({
  attribute: "heading-level",
  mode: "fromView",
  converter: nullableNumberConverter
})], AccordionItem.prototype, "headinglevel", void 0);

__decorate$1([attr({
  mode: "boolean"
})], AccordionItem.prototype, "expanded", void 0);

__decorate$1([attr], AccordionItem.prototype, "id", void 0);

applyMixins(AccordionItem, StartEnd);

/**
 * The template for the {@link @microsoft/fast-foundation#Accordion} component.
 * @public
 */

const accordionTemplate = (context, definition) =>
/* TODO: deprecate slot name `item` to only support default slot https://github.com/microsoft/fast/issues/5515 */
html`<template><slot ${slotted({
  property: "accordionItems",
  filter: elements()
})}></slot><slot name="item" part="item" ${slotted("accordionItems")}></slot></template>`;

/**
 * Standard orientation values
 */
const Orientation = {
  horizontal: "horizontal",
  vertical: "vertical"
};

/**
 * Returns the index of the last element in the array where predicate is true, and -1 otherwise.
 *
 * @param array - the array to test
 * @param predicate - find calls predicate once for each element of the array, in descending order, until it finds one where predicate returns true. If such an element is found, findLastIndex immediately returns that element index. Otherwise, findIndex returns -1.
 */
function findLastIndex(array, predicate) {
  let k = array.length;

  while (k--) {
    if (predicate(array[k], k, array)) {
      return k;
    }
  }

  return -1;
}

/**
 * Checks if the DOM is available to access and use
 */
function canUseDOM() {
  return !!(typeof window !== "undefined" && window.document && window.document.createElement);
}

/**
 * A test that ensures that all arguments are HTML Elements
 */

function isHTMLElement(...args) {
  return args.every(arg => arg instanceof HTMLElement);
}
/**
 * Returns all displayed elements inside of a root node that match a provided selector
 */

function getDisplayedNodes(rootNode, selector) {
  if (!rootNode || !selector || !isHTMLElement(rootNode)) {
    return;
  }

  const nodes = Array.from(rootNode.querySelectorAll(selector)); // offsetParent will be null if the element isn't currently displayed,
  // so this will allow us to operate only on visible nodes

  return nodes.filter(node => node.offsetParent !== null);
}
/**
 * Returns the nonce used in the page, if any.
 *
 * Based on https://github.com/cssinjs/jss/blob/master/packages/jss/src/DomRenderer.js
 */

function getNonce() {
  const node = document.querySelector('meta[property="csp-nonce"]');

  if (node) {
    return node.getAttribute("content");
  } else {
    return null;
  }
}
/**
 * Test if the document supports :focus-visible
 */


let _canUseFocusVisible;

function canUseFocusVisible() {
  if (typeof _canUseFocusVisible === "boolean") {
    return _canUseFocusVisible;
  }

  if (!canUseDOM()) {
    _canUseFocusVisible = false;
    return _canUseFocusVisible;
  } // Check to see if the document supports the focus-visible element


  const styleElement = document.createElement("style"); // If nonces are present on the page, use it when creating the style element
  // to test focus-visible support.

  const styleNonce = getNonce();

  if (styleNonce !== null) {
    styleElement.setAttribute("nonce", styleNonce);
  }

  document.head.appendChild(styleElement);

  try {
    styleElement.sheet.insertRule("foo:focus-visible {color:inherit}", 0);
    _canUseFocusVisible = true;
  } catch (e) {
    _canUseFocusVisible = false;
  } finally {
    document.head.removeChild(styleElement);
  }

  return _canUseFocusVisible;
}

/**
 * This set of exported strings reference https://developer.mozilla.org/en-US/docs/Web/Events
 * and should include all non-deprecated and non-experimental Standard events
 */
const eventFocus = "focus";
const eventFocusIn = "focusin";
const eventFocusOut = "focusout";
const eventKeyDown = "keydown";
const eventResize = "resize";
const eventScroll = "scroll";

/**
 * Key Code values
 * @deprecated - KeyCodes are deprecated, use individual string key exports
 */
var KeyCodes;

(function (KeyCodes) {
  KeyCodes[KeyCodes["alt"] = 18] = "alt";
  KeyCodes[KeyCodes["arrowDown"] = 40] = "arrowDown";
  KeyCodes[KeyCodes["arrowLeft"] = 37] = "arrowLeft";
  KeyCodes[KeyCodes["arrowRight"] = 39] = "arrowRight";
  KeyCodes[KeyCodes["arrowUp"] = 38] = "arrowUp";
  KeyCodes[KeyCodes["back"] = 8] = "back";
  KeyCodes[KeyCodes["backSlash"] = 220] = "backSlash";
  KeyCodes[KeyCodes["break"] = 19] = "break";
  KeyCodes[KeyCodes["capsLock"] = 20] = "capsLock";
  KeyCodes[KeyCodes["closeBracket"] = 221] = "closeBracket";
  KeyCodes[KeyCodes["colon"] = 186] = "colon";
  KeyCodes[KeyCodes["colon2"] = 59] = "colon2";
  KeyCodes[KeyCodes["comma"] = 188] = "comma";
  KeyCodes[KeyCodes["ctrl"] = 17] = "ctrl";
  KeyCodes[KeyCodes["delete"] = 46] = "delete";
  KeyCodes[KeyCodes["end"] = 35] = "end";
  KeyCodes[KeyCodes["enter"] = 13] = "enter";
  KeyCodes[KeyCodes["equals"] = 187] = "equals";
  KeyCodes[KeyCodes["equals2"] = 61] = "equals2";
  KeyCodes[KeyCodes["equals3"] = 107] = "equals3";
  KeyCodes[KeyCodes["escape"] = 27] = "escape";
  KeyCodes[KeyCodes["forwardSlash"] = 191] = "forwardSlash";
  KeyCodes[KeyCodes["function1"] = 112] = "function1";
  KeyCodes[KeyCodes["function10"] = 121] = "function10";
  KeyCodes[KeyCodes["function11"] = 122] = "function11";
  KeyCodes[KeyCodes["function12"] = 123] = "function12";
  KeyCodes[KeyCodes["function2"] = 113] = "function2";
  KeyCodes[KeyCodes["function3"] = 114] = "function3";
  KeyCodes[KeyCodes["function4"] = 115] = "function4";
  KeyCodes[KeyCodes["function5"] = 116] = "function5";
  KeyCodes[KeyCodes["function6"] = 117] = "function6";
  KeyCodes[KeyCodes["function7"] = 118] = "function7";
  KeyCodes[KeyCodes["function8"] = 119] = "function8";
  KeyCodes[KeyCodes["function9"] = 120] = "function9";
  KeyCodes[KeyCodes["home"] = 36] = "home";
  KeyCodes[KeyCodes["insert"] = 45] = "insert";
  KeyCodes[KeyCodes["menu"] = 93] = "menu";
  KeyCodes[KeyCodes["minus"] = 189] = "minus";
  KeyCodes[KeyCodes["minus2"] = 109] = "minus2";
  KeyCodes[KeyCodes["numLock"] = 144] = "numLock";
  KeyCodes[KeyCodes["numPad0"] = 96] = "numPad0";
  KeyCodes[KeyCodes["numPad1"] = 97] = "numPad1";
  KeyCodes[KeyCodes["numPad2"] = 98] = "numPad2";
  KeyCodes[KeyCodes["numPad3"] = 99] = "numPad3";
  KeyCodes[KeyCodes["numPad4"] = 100] = "numPad4";
  KeyCodes[KeyCodes["numPad5"] = 101] = "numPad5";
  KeyCodes[KeyCodes["numPad6"] = 102] = "numPad6";
  KeyCodes[KeyCodes["numPad7"] = 103] = "numPad7";
  KeyCodes[KeyCodes["numPad8"] = 104] = "numPad8";
  KeyCodes[KeyCodes["numPad9"] = 105] = "numPad9";
  KeyCodes[KeyCodes["numPadDivide"] = 111] = "numPadDivide";
  KeyCodes[KeyCodes["numPadDot"] = 110] = "numPadDot";
  KeyCodes[KeyCodes["numPadMinus"] = 109] = "numPadMinus";
  KeyCodes[KeyCodes["numPadMultiply"] = 106] = "numPadMultiply";
  KeyCodes[KeyCodes["numPadPlus"] = 107] = "numPadPlus";
  KeyCodes[KeyCodes["openBracket"] = 219] = "openBracket";
  KeyCodes[KeyCodes["pageDown"] = 34] = "pageDown";
  KeyCodes[KeyCodes["pageUp"] = 33] = "pageUp";
  KeyCodes[KeyCodes["period"] = 190] = "period";
  KeyCodes[KeyCodes["print"] = 44] = "print";
  KeyCodes[KeyCodes["quote"] = 222] = "quote";
  KeyCodes[KeyCodes["scrollLock"] = 145] = "scrollLock";
  KeyCodes[KeyCodes["shift"] = 16] = "shift";
  KeyCodes[KeyCodes["space"] = 32] = "space";
  KeyCodes[KeyCodes["tab"] = 9] = "tab";
  KeyCodes[KeyCodes["tilde"] = 192] = "tilde";
  KeyCodes[KeyCodes["windowsLeft"] = 91] = "windowsLeft";
  KeyCodes[KeyCodes["windowsOpera"] = 219] = "windowsOpera";
  KeyCodes[KeyCodes["windowsRight"] = 92] = "windowsRight";
})(KeyCodes || (KeyCodes = {}));
/**
 * String values for use with KeyboardEvent.key
 */

const keyArrowDown = "ArrowDown";
const keyArrowLeft = "ArrowLeft";
const keyArrowRight = "ArrowRight";
const keyArrowUp = "ArrowUp";
const keyEnter = "Enter";
const keyEscape = "Escape";
const keyHome = "Home";
const keyEnd = "End";
const keyFunction2 = "F2";
const keyPageDown = "PageDown";
const keyPageUp = "PageUp";
const keySpace = " ";
const keyTab = "Tab";
const keyBackspace = "Backspace";
const keyDelete = "Delete";
const ArrowKeys = {
  ArrowDown: keyArrowDown,
  ArrowLeft: keyArrowLeft,
  ArrowRight: keyArrowRight,
  ArrowUp: keyArrowUp
};

/**
 * Expose ltr and rtl strings
 */
var Direction;

(function (Direction) {
  Direction["ltr"] = "ltr";
  Direction["rtl"] = "rtl";
})(Direction || (Direction = {}));

/**
 * This method keeps a given value within the bounds of a min and max value. If the value
 * is larger than the max, the minimum value will be returned. If the value is smaller than the minimum,
 * the maximum will be returned. Otherwise, the value is returned un-changed.
 */
function wrapInBounds(min, max, value) {
  if (value < min) {
    return max;
  } else if (value > max) {
    return min;
  }

  return value;
}
/**
 * Ensures that a value is between a min and max value. If value is lower than min, min will be returned.
 * If value is greater than max, max will be returned.
 */

function limit(min, max, value) {
  return Math.min(Math.max(value, min), max);
}
/**
 * Determines if a number value is within a specified range.
 *
 * @param value - the value to check
 * @param min - the range start
 * @param max - the range end
 */

function inRange(value, min, max = 0) {
  [min, max] = [min, max].sort((a, b) => a - b);
  return min <= value && value < max;
}

let uniqueIdCounter = 0;
/**
 * Generates a unique ID based on incrementing a counter.
 */

function uniqueId(prefix = "") {
  return `${prefix}${uniqueIdCounter++}`;
}

/**
 * Define system colors for use in CSS stylesheets.
 *
 * https://drafts.csswg.org/css-color/#css-system-colors
 */
var SystemColors;

(function (SystemColors) {
  SystemColors["Canvas"] = "Canvas";
  SystemColors["CanvasText"] = "CanvasText";
  SystemColors["LinkText"] = "LinkText";
  SystemColors["VisitedText"] = "VisitedText";
  SystemColors["ActiveText"] = "ActiveText";
  SystemColors["ButtonFace"] = "ButtonFace";
  SystemColors["ButtonText"] = "ButtonText";
  SystemColors["Field"] = "Field";
  SystemColors["FieldText"] = "FieldText";
  SystemColors["Highlight"] = "Highlight";
  SystemColors["HighlightText"] = "HighlightText";
  SystemColors["GrayText"] = "GrayText";
})(SystemColors || (SystemColors = {}));

/**
 * Expand mode for {@link Accordion}
 * @public
 */

const AccordionExpandMode = {
  /**
   * Designates only a single {@link @microsoft/fast-foundation#(AccordionItem:class) } can be open a time.
   */
  single: "single",

  /**
   * Designates multiple {@link @microsoft/fast-foundation#(AccordionItem:class) | AccordionItems} can be open simultaneously.
   */
  multi: "multi"
};
/**
 * An Accordion Custom HTML Element
 * Implements {@link https://www.w3.org/TR/wai-aria-practices-1.1/#accordion | ARIA Accordion}.
 *
 * @fires change - Fires a custom 'change' event when the active item changes
 * @csspart item - The slot for the accordion items
 * @public
 *
 * @remarks
 * Designed to be used with {@link @microsoft/fast-foundation#accordionTemplate} and {@link @microsoft/fast-foundation#(AccordionItem:class)}.
 */

class Accordion extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * Controls the expand mode of the Accordion, either allowing
     * single or multiple item expansion.
     * @public
     *
     * @remarks
     * HTML attribute: expand-mode
     */

    this.expandmode = AccordionExpandMode.multi;
    this.activeItemIndex = 0;

    this.change = () => {
      this.$emit("change");
    };

    this.setItems = () => {
      var _a;

      if (this.accordionItems.length === 0) {
        return;
      }

      this.accordionIds = this.getItemIds();
      this.accordionItems.forEach((item, index) => {
        if (item instanceof AccordionItem) {
          item.addEventListener("change", this.activeItemChange);

          if (this.isSingleExpandMode()) {
            this.activeItemIndex !== index ? item.expanded = false : item.expanded = true;
          }
        }

        const itemId = this.accordionIds[index];
        item.setAttribute("id", typeof itemId !== "string" ? `accordion-${index + 1}` : itemId);
        this.activeid = this.accordionIds[this.activeItemIndex];
        item.addEventListener("keydown", this.handleItemKeyDown);
        item.addEventListener("focus", this.handleItemFocus);
      });

      if (this.isSingleExpandMode()) {
        const expandedItem = (_a = this.findExpandedItem()) !== null && _a !== void 0 ? _a : this.accordionItems[0];
        expandedItem.setAttribute("aria-disabled", "true");
      }
    };

    this.removeItemListeners = oldValue => {
      oldValue.forEach((item, index) => {
        item.removeEventListener("change", this.activeItemChange);
        item.removeEventListener("keydown", this.handleItemKeyDown);
        item.removeEventListener("focus", this.handleItemFocus);
      });
    };

    this.activeItemChange = event => {
      const selectedItem = event.target;
      this.activeid = selectedItem.getAttribute("id");

      if (this.isSingleExpandMode()) {
        this.resetItems();
        selectedItem.expanded = true;
        selectedItem.setAttribute("aria-disabled", "true");
        this.accordionItems.forEach(item => {
          if (!item.hasAttribute("disabled") && item.id !== this.activeid) {
            item.removeAttribute("aria-disabled");
          }
        });
      }

      this.activeItemIndex = Array.from(this.accordionItems).indexOf(selectedItem);
      this.change();
    };

    this.handleItemKeyDown = event => {
      // only handle the keydown if the event target is the accordion item
      // prevents arrow keys from moving focus to accordion headers when focus is on accordion item panel content
      if (event.target !== event.currentTarget) {
        return;
      }

      this.accordionIds = this.getItemIds();

      switch (event.key) {
        case keyArrowUp:
          event.preventDefault();
          this.adjust(-1);
          break;

        case keyArrowDown:
          event.preventDefault();
          this.adjust(1);
          break;

        case keyHome:
          this.activeItemIndex = 0;
          this.focusItem();
          break;

        case keyEnd:
          this.activeItemIndex = this.accordionItems.length - 1;
          this.focusItem();
          break;
      }
    };

    this.handleItemFocus = event => {
      // update the active item index if the focus moves to an accordion item via a different method other than the up and down arrow key actions
      // only do so if the focus is actually on the accordion item and not on any of its children
      if (event.target === event.currentTarget) {
        const focusedItem = event.target;
        const focusedIndex = this.activeItemIndex = Array.from(this.accordionItems).indexOf(focusedItem);

        if (this.activeItemIndex !== focusedIndex && focusedIndex !== -1) {
          this.activeItemIndex = focusedIndex;
          this.activeid = this.accordionIds[this.activeItemIndex];
        }
      }
    };
  }
  /**
   * @internal
   */


  accordionItemsChanged(oldValue, newValue) {
    if (this.$fastController.isConnected) {
      this.removeItemListeners(oldValue);
      this.setItems();
    }
  }

  findExpandedItem() {
    for (let item = 0; item < this.accordionItems.length; item++) {
      if (this.accordionItems[item].getAttribute("expanded") === "true") {
        return this.accordionItems[item];
      }
    }

    return null;
  }

  resetItems() {
    this.accordionItems.forEach((item, index) => {
      item.expanded = false;
    });
  }

  getItemIds() {
    return this.accordionItems.map(accordionItem => {
      return accordionItem.getAttribute("id");
    });
  }

  isSingleExpandMode() {
    return this.expandmode === AccordionExpandMode.single;
  }

  adjust(adjustment) {
    this.activeItemIndex = wrapInBounds(0, this.accordionItems.length - 1, this.activeItemIndex + adjustment);
    this.focusItem();
  }

  focusItem() {
    const element = this.accordionItems[this.activeItemIndex];

    if (element instanceof AccordionItem) {
      element.expandbutton.focus();
    }
  }

}

__decorate$1([attr({
  attribute: "expand-mode"
})], Accordion.prototype, "expandmode", void 0);

__decorate$1([observable], Accordion.prototype, "accordionItems", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(Anchor:class)} component.
 * @public
 */

const anchorTemplate = (context, definition) => html`<a class="control" part="control" download="${x => x.download}" href="${x => x.href}" hreflang="${x => x.hreflang}" ping="${x => x.ping}" referrerpolicy="${x => x.referrerpolicy}" rel="${x => x.rel}" target="${x => x.target}" type="${x => x.type}" aria-atomic="${x => x.ariaAtomic}" aria-busy="${x => x.ariaBusy}" aria-controls="${x => x.ariaControls}" aria-current="${x => x.ariaCurrent}" aria-describedby="${x => x.ariaDescribedby}" aria-details="${x => x.ariaDetails}" aria-disabled="${x => x.ariaDisabled}" aria-errormessage="${x => x.ariaErrormessage}" aria-expanded="${x => x.ariaExpanded}" aria-flowto="${x => x.ariaFlowto}" aria-haspopup="${x => x.ariaHaspopup}" aria-hidden="${x => x.ariaHidden}" aria-invalid="${x => x.ariaInvalid}" aria-keyshortcuts="${x => x.ariaKeyshortcuts}" aria-label="${x => x.ariaLabel}" aria-labelledby="${x => x.ariaLabelledby}" aria-live="${x => x.ariaLive}" aria-owns="${x => x.ariaOwns}" aria-relevant="${x => x.ariaRelevant}" aria-roledescription="${x => x.ariaRoledescription}" ${ref("control")}>${startSlotTemplate(context, definition)}<span class="content" part="content"><slot ${slotted("defaultSlottedContent")}></slot></span>${endSlotTemplate(context, definition)}</a>`;

/**
 * Some states and properties are applicable to all host language elements regardless of whether a role is applied.
 * The following global states and properties are supported by all roles and by all base markup elements.
 * {@link https://www.w3.org/TR/wai-aria-1.1/#global_states}
 *
 * This is intended to be used as a mixin. Be sure you extend FASTElement.
 *
 * @public
 */

class ARIAGlobalStatesAndProperties {}

__decorate$1([attr({
  attribute: "aria-atomic"
})], ARIAGlobalStatesAndProperties.prototype, "ariaAtomic", void 0);

__decorate$1([attr({
  attribute: "aria-busy"
})], ARIAGlobalStatesAndProperties.prototype, "ariaBusy", void 0);

__decorate$1([attr({
  attribute: "aria-controls"
})], ARIAGlobalStatesAndProperties.prototype, "ariaControls", void 0);

__decorate$1([attr({
  attribute: "aria-current"
})], ARIAGlobalStatesAndProperties.prototype, "ariaCurrent", void 0);

__decorate$1([attr({
  attribute: "aria-describedby"
})], ARIAGlobalStatesAndProperties.prototype, "ariaDescribedby", void 0);

__decorate$1([attr({
  attribute: "aria-details"
})], ARIAGlobalStatesAndProperties.prototype, "ariaDetails", void 0);

__decorate$1([attr({
  attribute: "aria-disabled"
})], ARIAGlobalStatesAndProperties.prototype, "ariaDisabled", void 0);

__decorate$1([attr({
  attribute: "aria-errormessage"
})], ARIAGlobalStatesAndProperties.prototype, "ariaErrormessage", void 0);

__decorate$1([attr({
  attribute: "aria-flowto"
})], ARIAGlobalStatesAndProperties.prototype, "ariaFlowto", void 0);

__decorate$1([attr({
  attribute: "aria-haspopup"
})], ARIAGlobalStatesAndProperties.prototype, "ariaHaspopup", void 0);

__decorate$1([attr({
  attribute: "aria-hidden"
})], ARIAGlobalStatesAndProperties.prototype, "ariaHidden", void 0);

__decorate$1([attr({
  attribute: "aria-invalid"
})], ARIAGlobalStatesAndProperties.prototype, "ariaInvalid", void 0);

__decorate$1([attr({
  attribute: "aria-keyshortcuts"
})], ARIAGlobalStatesAndProperties.prototype, "ariaKeyshortcuts", void 0);

__decorate$1([attr({
  attribute: "aria-label"
})], ARIAGlobalStatesAndProperties.prototype, "ariaLabel", void 0);

__decorate$1([attr({
  attribute: "aria-labelledby"
})], ARIAGlobalStatesAndProperties.prototype, "ariaLabelledby", void 0);

__decorate$1([attr({
  attribute: "aria-live"
})], ARIAGlobalStatesAndProperties.prototype, "ariaLive", void 0);

__decorate$1([attr({
  attribute: "aria-owns"
})], ARIAGlobalStatesAndProperties.prototype, "ariaOwns", void 0);

__decorate$1([attr({
  attribute: "aria-relevant"
})], ARIAGlobalStatesAndProperties.prototype, "ariaRelevant", void 0);

__decorate$1([attr({
  attribute: "aria-roledescription"
})], ARIAGlobalStatesAndProperties.prototype, "ariaRoledescription", void 0);

/**
 * An Anchor Custom HTML Element.
 * Based largely on the {@link https://developer.mozilla.org/en-US/docs/Web/HTML/Element/a | <a> element }.
 *
 * @slot start - Content which can be provided before the anchor content
 * @slot end - Content which can be provided after the anchor content
 * @slot - The default slot for anchor content
 * @csspart control - The anchor element
 * @csspart content - The element wrapping anchor content
 *
 * @public
 */

class Anchor$1 extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * Overrides the focus call for where delegatesFocus is unsupported.
     * This check works for Chrome, Edge Chromium, FireFox, and Safari
     * Relevant PR on the Firefox browser: https://phabricator.services.mozilla.com/D123858
     */

    this.handleUnsupportedDelegatesFocus = () => {
      var _a; // Check to see if delegatesFocus is supported


      if (window.ShadowRoot && !window.ShadowRoot.prototype.hasOwnProperty("delegatesFocus") && ((_a = this.$fastController.definition.shadowOptions) === null || _a === void 0 ? void 0 : _a.delegatesFocus)) {
        this.focus = () => {
          this.control.focus();
        };
      }
    };
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.handleUnsupportedDelegatesFocus();
  }

}

__decorate$1([attr], Anchor$1.prototype, "download", void 0);

__decorate$1([attr], Anchor$1.prototype, "href", void 0);

__decorate$1([attr], Anchor$1.prototype, "hreflang", void 0);

__decorate$1([attr], Anchor$1.prototype, "ping", void 0);

__decorate$1([attr], Anchor$1.prototype, "referrerpolicy", void 0);

__decorate$1([attr], Anchor$1.prototype, "rel", void 0);

__decorate$1([attr], Anchor$1.prototype, "target", void 0);

__decorate$1([attr], Anchor$1.prototype, "type", void 0);

__decorate$1([observable], Anchor$1.prototype, "defaultSlottedContent", void 0);
/**
 * Includes ARIA states and properties relating to the ARIA link role
 *
 * @public
 */


class DelegatesARIALink {}

__decorate$1([attr({
  attribute: "aria-expanded"
})], DelegatesARIALink.prototype, "ariaExpanded", void 0);

applyMixins(DelegatesARIALink, ARIAGlobalStatesAndProperties);
applyMixins(Anchor$1, StartEnd, DelegatesARIALink);

/**
 * The template for the {@link @microsoft/fast-foundation#(AnchoredRegion:class)} component.
 * @public
 */

const anchoredRegionTemplate = (context, definition) => html`<template class="${x => x.initialLayoutComplete ? "loaded" : ""}">${when(x => x.initialLayoutComplete, html`<slot></slot>`)}</template>`;

/**
 * a method to determine the current localization direction of the view
 * @param rootNode - the HTMLElement to begin the query from, usually "this" when used in a component controller
 * @public
 */

const getDirection = rootNode => {
  const dirNode = rootNode.closest("[dir]");
  return dirNode !== null && dirNode.dir === "rtl" ? Direction.rtl : Direction.ltr;
};

/**
 *  A service to batch intersection event callbacks so multiple elements can share a single observer
 *
 * @public
 */

class IntersectionService {
  constructor() {
    this.intersectionDetector = null;
    this.observedElements = new Map();
    /**
     * Request the position of a target element
     *
     * @internal
     */

    this.requestPosition = (target, callback) => {
      var _a;

      if (this.intersectionDetector === null) {
        return;
      }

      if (this.observedElements.has(target)) {
        (_a = this.observedElements.get(target)) === null || _a === void 0 ? void 0 : _a.push(callback);
        return;
      }

      this.observedElements.set(target, [callback]);
      this.intersectionDetector.observe(target);
    };
    /**
     * Cancel a position request
     *
     * @internal
     */


    this.cancelRequestPosition = (target, callback) => {
      const callbacks = this.observedElements.get(target);

      if (callbacks !== undefined) {
        const callBackIndex = callbacks.indexOf(callback);

        if (callBackIndex !== -1) {
          callbacks.splice(callBackIndex, 1);
        }
      }
    };
    /**
     * initialize intersection detector
     */


    this.initializeIntersectionDetector = () => {
      if (!$global.IntersectionObserver) {
        //intersection observer not supported
        return;
      }

      this.intersectionDetector = new IntersectionObserver(this.handleIntersection, {
        root: null,
        rootMargin: "0px",
        threshold: [0, 1]
      });
    };
    /**
     *  Handle intersections
     */


    this.handleIntersection = entries => {
      if (this.intersectionDetector === null) {
        return;
      }

      const pendingCallbacks = [];
      const pendingCallbackParams = []; // go through the entries to build a list of callbacks and params for each

      entries.forEach(entry => {
        var _a; // stop watching this element until we get new update requests for it


        (_a = this.intersectionDetector) === null || _a === void 0 ? void 0 : _a.unobserve(entry.target);
        const thisElementCallbacks = this.observedElements.get(entry.target);

        if (thisElementCallbacks !== undefined) {
          thisElementCallbacks.forEach(callback => {
            let targetCallbackIndex = pendingCallbacks.indexOf(callback);

            if (targetCallbackIndex === -1) {
              targetCallbackIndex = pendingCallbacks.length;
              pendingCallbacks.push(callback);
              pendingCallbackParams.push([]);
            }

            pendingCallbackParams[targetCallbackIndex].push(entry);
          });
          this.observedElements.delete(entry.target);
        }
      }); // execute callbacks

      pendingCallbacks.forEach((callback, index) => {
        callback(pendingCallbackParams[index]);
      });
    };

    this.initializeIntersectionDetector();
  }

}

/**
 * An anchored region Custom HTML Element.
 *
 * @slot - The default slot for the content
 * @fires loaded - Fires a custom 'loaded' event when the region is loaded and visible
 * @fires positionchange - Fires a custom 'positionchange' event when the position has changed
 *
 * @public
 */

class AnchoredRegion extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * The HTML ID of the anchor element this region is positioned relative to
     *
     * @public
     * @remarks
     * HTML Attribute: anchor
     */

    this.anchor = "";
    /**
     * The HTML ID of the viewport element this region is positioned relative to
     *
     * @public
     * @remarks
     * HTML Attribute: anchor
     */

    this.viewport = "";
    /**
     * Sets what logic the component uses to determine horizontal placement.
     * 'locktodefault' forces the default position
     * 'dynamic' decides placement based on available space
     * 'uncontrolled' does not control placement on the horizontal axis
     *
     * @public
     * @remarks
     * HTML Attribute: horizontal-positioning-mode
     */

    this.horizontalPositioningMode = "uncontrolled";
    /**
     * The default horizontal position of the region relative to the anchor element
     *
     * @public
     * @remarks
     * HTML Attribute: horizontal-default-position
     */

    this.horizontalDefaultPosition = "unset";
    /**
     * Whether the region remains in the viewport (ie. detaches from the anchor) on the horizontal axis
     *
     * @public
     * @remarks
     * HTML Attribute: horizontal-viewport-lock
     */

    this.horizontalViewportLock = false;
    /**
     * Whether the region overlaps the anchor on the horizontal axis
     *
     * @public
     * @remarks
     * HTML Attribute: horizontal-inset
     */

    this.horizontalInset = false;
    /**
     * Defines how the width of the region is calculated
     *
     * @public
     * @remarks
     * HTML Attribute: horizontal-scaling
     */

    this.horizontalScaling = "content";
    /**
     * Sets what logic the component uses to determine vertical placement.
     * 'locktodefault' forces the default position
     * 'dynamic' decides placement based on available space
     * 'uncontrolled' does not control placement on the vertical axis
     *
     * @public
     * @remarks
     * HTML Attribute: vertical-positioning-mode
     */

    this.verticalPositioningMode = "uncontrolled";
    /**
     * The default vertical position of the region relative to the anchor element
     *
     * @public
     * @remarks
     * HTML Attribute: vertical-default-position
     */

    this.verticalDefaultPosition = "unset";
    /**
     * Whether the region remains in the viewport (ie. detaches from the anchor) on the vertical axis
     *
     * @public
     * @remarks
     * HTML Attribute: vertical-viewport-lock
     */

    this.verticalViewportLock = false;
    /**
     * Whether the region overlaps the anchor on the vertical axis
     *
     * @public
     * @remarks
     * HTML Attribute: vertical-inset
     */

    this.verticalInset = false;
    /**
     * Defines how the height of the region is calculated
     *
     * @public
     * @remarks
     * HTML Attribute: vertical-scaling
     */

    this.verticalScaling = "content";
    /**
     * Whether the region is positioned using css "position: fixed".
     * Otherwise the region uses "position: absolute".
     * Fixed placement allows the region to break out of parent containers,
     *
     * @public
     * @remarks
     * HTML Attribute: fixed-placement
     */

    this.fixedPlacement = false;
    /**
     * Defines what triggers the anchored region to revaluate positioning
     *
     * @public
     * @remarks
     * HTML Attribute: auto-update-mode
     */

    this.autoUpdateMode = "anchor";
    /**
     * The HTML element being used as the anchor
     *
     * @public
     */

    this.anchorElement = null;
    /**
     * The HTML element being used as the viewport
     *
     * @public
     */

    this.viewportElement = null;
    /**
     * indicates that an initial positioning pass on layout has completed
     *
     * @internal
     */

    this.initialLayoutComplete = false;
    this.resizeDetector = null;
    /**
     * base offsets between the positioner's base position and the anchor's
     */

    this.baseHorizontalOffset = 0;
    this.baseVerticalOffset = 0;
    this.pendingPositioningUpdate = false;
    this.pendingReset = false;
    this.currentDirection = Direction.ltr;
    this.regionVisible = false; // indicates that a layout update should occur even if geometry has not changed
    // used to ensure some attribute changes are applied

    this.forceUpdate = false; // defines how big a difference in pixels there must be between states to
    // justify a layout update that affects the dom (prevents repeated sub-pixel corrections)

    this.updateThreshold = 0.5;
    /**
     * update position
     */

    this.update = () => {
      if (!this.pendingPositioningUpdate) {
        this.requestPositionUpdates();
      }
    };
    /**
     * starts observers
     */


    this.startObservers = () => {
      this.stopObservers();

      if (this.anchorElement === null) {
        return;
      }

      this.requestPositionUpdates();

      if (this.resizeDetector !== null) {
        this.resizeDetector.observe(this.anchorElement);
        this.resizeDetector.observe(this);
      }
    };
    /**
     * get position updates
     */


    this.requestPositionUpdates = () => {
      if (this.anchorElement === null || this.pendingPositioningUpdate) {
        return;
      }

      AnchoredRegion.intersectionService.requestPosition(this, this.handleIntersection);
      AnchoredRegion.intersectionService.requestPosition(this.anchorElement, this.handleIntersection);

      if (this.viewportElement !== null) {
        AnchoredRegion.intersectionService.requestPosition(this.viewportElement, this.handleIntersection);
      }

      this.pendingPositioningUpdate = true;
    };
    /**
     * stops observers
     */


    this.stopObservers = () => {
      if (this.pendingPositioningUpdate) {
        this.pendingPositioningUpdate = false;
        AnchoredRegion.intersectionService.cancelRequestPosition(this, this.handleIntersection);

        if (this.anchorElement !== null) {
          AnchoredRegion.intersectionService.cancelRequestPosition(this.anchorElement, this.handleIntersection);
        }

        if (this.viewportElement !== null) {
          AnchoredRegion.intersectionService.cancelRequestPosition(this.viewportElement, this.handleIntersection);
        }
      }

      if (this.resizeDetector !== null) {
        this.resizeDetector.disconnect();
      }
    };
    /**
     * Gets the viewport element by id, or defaults to document root
     */


    this.getViewport = () => {
      if (typeof this.viewport !== "string" || this.viewport === "") {
        return document.documentElement;
      }

      return document.getElementById(this.viewport);
    };
    /**
     *  Gets the anchor element by id
     */


    this.getAnchor = () => {
      return document.getElementById(this.anchor);
    };
    /**
     *  Handle intersections
     */


    this.handleIntersection = entries => {
      if (!this.pendingPositioningUpdate) {
        return;
      }

      this.pendingPositioningUpdate = false;

      if (!this.applyIntersectionEntries(entries)) {
        return;
      }

      this.updateLayout();
    };
    /**
     *  iterate through intersection entries and apply data
     */


    this.applyIntersectionEntries = entries => {
      const regionEntry = entries.find(x => x.target === this);
      const anchorEntry = entries.find(x => x.target === this.anchorElement);
      const viewportEntry = entries.find(x => x.target === this.viewportElement);

      if (regionEntry === undefined || viewportEntry === undefined || anchorEntry === undefined) {
        return false;
      } // don't update the dom unless there is a significant difference in rect positions


      if (!this.regionVisible || this.forceUpdate || this.regionRect === undefined || this.anchorRect === undefined || this.viewportRect === undefined || this.isRectDifferent(this.anchorRect, anchorEntry.boundingClientRect) || this.isRectDifferent(this.viewportRect, viewportEntry.boundingClientRect) || this.isRectDifferent(this.regionRect, regionEntry.boundingClientRect)) {
        this.regionRect = regionEntry.boundingClientRect;
        this.anchorRect = anchorEntry.boundingClientRect;

        if (this.viewportElement === document.documentElement) {
          this.viewportRect = new DOMRectReadOnly(viewportEntry.boundingClientRect.x + document.documentElement.scrollLeft, viewportEntry.boundingClientRect.y + document.documentElement.scrollTop, viewportEntry.boundingClientRect.width, viewportEntry.boundingClientRect.height);
        } else {
          this.viewportRect = viewportEntry.boundingClientRect;
        }

        this.updateRegionOffset();
        this.forceUpdate = false;
        return true;
      }

      return false;
    };
    /**
     *  Update the offset values
     */


    this.updateRegionOffset = () => {
      if (this.anchorRect && this.regionRect) {
        this.baseHorizontalOffset = this.baseHorizontalOffset + (this.anchorRect.left - this.regionRect.left) + (this.translateX - this.baseHorizontalOffset);
        this.baseVerticalOffset = this.baseVerticalOffset + (this.anchorRect.top - this.regionRect.top) + (this.translateY - this.baseVerticalOffset);
      }
    };
    /**
     *  compare rects to see if there is enough change to justify a DOM update
     */


    this.isRectDifferent = (rectA, rectB) => {
      if (Math.abs(rectA.top - rectB.top) > this.updateThreshold || Math.abs(rectA.right - rectB.right) > this.updateThreshold || Math.abs(rectA.bottom - rectB.bottom) > this.updateThreshold || Math.abs(rectA.left - rectB.left) > this.updateThreshold) {
        return true;
      }

      return false;
    };
    /**
     *  Handle resize events
     */


    this.handleResize = entries => {
      this.update();
    };
    /**
     * resets the component
     */


    this.reset = () => {
      if (!this.pendingReset) {
        return;
      }

      this.pendingReset = false;

      if (this.anchorElement === null) {
        this.anchorElement = this.getAnchor();
      }

      if (this.viewportElement === null) {
        this.viewportElement = this.getViewport();
      }

      this.currentDirection = getDirection(this);
      this.startObservers();
    };
    /**
     *  Recalculate layout related state values
     */


    this.updateLayout = () => {
      let desiredVerticalPosition = undefined;
      let desiredHorizontalPosition = undefined;

      if (this.horizontalPositioningMode !== "uncontrolled") {
        const horizontalOptions = this.getPositioningOptions(this.horizontalInset);

        if (this.horizontalDefaultPosition === "center") {
          desiredHorizontalPosition = "center";
        } else if (this.horizontalDefaultPosition !== "unset") {
          let dirCorrectedHorizontalDefaultPosition = this.horizontalDefaultPosition;

          if (dirCorrectedHorizontalDefaultPosition === "start" || dirCorrectedHorizontalDefaultPosition === "end") {
            // if direction changes we reset the layout
            const newDirection = getDirection(this);

            if (newDirection !== this.currentDirection) {
              this.currentDirection = newDirection;
              this.initialize();
              return;
            }

            if (this.currentDirection === Direction.ltr) {
              dirCorrectedHorizontalDefaultPosition = dirCorrectedHorizontalDefaultPosition === "start" ? "left" : "right";
            } else {
              dirCorrectedHorizontalDefaultPosition = dirCorrectedHorizontalDefaultPosition === "start" ? "right" : "left";
            }
          }

          switch (dirCorrectedHorizontalDefaultPosition) {
            case "left":
              desiredHorizontalPosition = this.horizontalInset ? "insetStart" : "start";
              break;

            case "right":
              desiredHorizontalPosition = this.horizontalInset ? "insetEnd" : "end";
              break;
          }
        }

        const horizontalThreshold = this.horizontalThreshold !== undefined ? this.horizontalThreshold : this.regionRect !== undefined ? this.regionRect.width : 0;
        const anchorLeft = this.anchorRect !== undefined ? this.anchorRect.left : 0;
        const anchorRight = this.anchorRect !== undefined ? this.anchorRect.right : 0;
        const anchorWidth = this.anchorRect !== undefined ? this.anchorRect.width : 0;
        const viewportLeft = this.viewportRect !== undefined ? this.viewportRect.left : 0;
        const viewportRight = this.viewportRect !== undefined ? this.viewportRect.right : 0;

        if (desiredHorizontalPosition === undefined || !(this.horizontalPositioningMode === "locktodefault") && this.getAvailableSpace(desiredHorizontalPosition, anchorLeft, anchorRight, anchorWidth, viewportLeft, viewportRight) < horizontalThreshold) {
          desiredHorizontalPosition = this.getAvailableSpace(horizontalOptions[0], anchorLeft, anchorRight, anchorWidth, viewportLeft, viewportRight) > this.getAvailableSpace(horizontalOptions[1], anchorLeft, anchorRight, anchorWidth, viewportLeft, viewportRight) ? horizontalOptions[0] : horizontalOptions[1];
        }
      }

      if (this.verticalPositioningMode !== "uncontrolled") {
        const verticalOptions = this.getPositioningOptions(this.verticalInset);

        if (this.verticalDefaultPosition === "center") {
          desiredVerticalPosition = "center";
        } else if (this.verticalDefaultPosition !== "unset") {
          switch (this.verticalDefaultPosition) {
            case "top":
              desiredVerticalPosition = this.verticalInset ? "insetStart" : "start";
              break;

            case "bottom":
              desiredVerticalPosition = this.verticalInset ? "insetEnd" : "end";
              break;
          }
        }

        const verticalThreshold = this.verticalThreshold !== undefined ? this.verticalThreshold : this.regionRect !== undefined ? this.regionRect.height : 0;
        const anchorTop = this.anchorRect !== undefined ? this.anchorRect.top : 0;
        const anchorBottom = this.anchorRect !== undefined ? this.anchorRect.bottom : 0;
        const anchorHeight = this.anchorRect !== undefined ? this.anchorRect.height : 0;
        const viewportTop = this.viewportRect !== undefined ? this.viewportRect.top : 0;
        const viewportBottom = this.viewportRect !== undefined ? this.viewportRect.bottom : 0;

        if (desiredVerticalPosition === undefined || !(this.verticalPositioningMode === "locktodefault") && this.getAvailableSpace(desiredVerticalPosition, anchorTop, anchorBottom, anchorHeight, viewportTop, viewportBottom) < verticalThreshold) {
          desiredVerticalPosition = this.getAvailableSpace(verticalOptions[0], anchorTop, anchorBottom, anchorHeight, viewportTop, viewportBottom) > this.getAvailableSpace(verticalOptions[1], anchorTop, anchorBottom, anchorHeight, viewportTop, viewportBottom) ? verticalOptions[0] : verticalOptions[1];
        }
      }

      const nextPositionerDimension = this.getNextRegionDimension(desiredHorizontalPosition, desiredVerticalPosition);
      const positionChanged = this.horizontalPosition !== desiredHorizontalPosition || this.verticalPosition !== desiredVerticalPosition;
      this.setHorizontalPosition(desiredHorizontalPosition, nextPositionerDimension);
      this.setVerticalPosition(desiredVerticalPosition, nextPositionerDimension);
      this.updateRegionStyle();

      if (!this.initialLayoutComplete) {
        this.initialLayoutComplete = true;
        this.requestPositionUpdates();
        return;
      }

      if (!this.regionVisible) {
        this.regionVisible = true;
        this.style.removeProperty("pointer-events");
        this.style.removeProperty("opacity");
        this.classList.toggle("loaded", true);
        this.$emit("loaded", this, {
          bubbles: false
        });
      }

      this.updatePositionClasses();

      if (positionChanged) {
        // emit change event
        this.$emit("positionchange", this, {
          bubbles: false
        });
      }
    };
    /**
     *  Updates the style string applied to the region element as well as the css classes attached
     *  to the root element
     */


    this.updateRegionStyle = () => {
      this.style.width = this.regionWidth;
      this.style.height = this.regionHeight;
      this.style.transform = `translate(${this.translateX}px, ${this.translateY}px)`;
    };
    /**
     *  Updates the css classes that reflect the current position of the element
     */


    this.updatePositionClasses = () => {
      this.classList.toggle("top", this.verticalPosition === "start");
      this.classList.toggle("bottom", this.verticalPosition === "end");
      this.classList.toggle("inset-top", this.verticalPosition === "insetStart");
      this.classList.toggle("inset-bottom", this.verticalPosition === "insetEnd");
      this.classList.toggle("vertical-center", this.verticalPosition === "center");
      this.classList.toggle("left", this.horizontalPosition === "start");
      this.classList.toggle("right", this.horizontalPosition === "end");
      this.classList.toggle("inset-left", this.horizontalPosition === "insetStart");
      this.classList.toggle("inset-right", this.horizontalPosition === "insetEnd");
      this.classList.toggle("horizontal-center", this.horizontalPosition === "center");
    };
    /**
     * Get horizontal positioning state based on desired position
     */


    this.setHorizontalPosition = (desiredHorizontalPosition, nextPositionerDimension) => {
      if (desiredHorizontalPosition === undefined || this.regionRect === undefined || this.anchorRect === undefined || this.viewportRect === undefined) {
        return;
      }

      let nextRegionWidth = 0;

      switch (this.horizontalScaling) {
        case "anchor":
        case "fill":
          nextRegionWidth = this.horizontalViewportLock ? this.viewportRect.width : nextPositionerDimension.width;
          this.regionWidth = `${nextRegionWidth}px`;
          break;

        case "content":
          nextRegionWidth = this.regionRect.width;
          this.regionWidth = "unset";
          break;
      }

      let sizeDelta = 0;

      switch (desiredHorizontalPosition) {
        case "start":
          this.translateX = this.baseHorizontalOffset - nextRegionWidth;

          if (this.horizontalViewportLock && this.anchorRect.left > this.viewportRect.right) {
            this.translateX = this.translateX - (this.anchorRect.left - this.viewportRect.right);
          }

          break;

        case "insetStart":
          this.translateX = this.baseHorizontalOffset - nextRegionWidth + this.anchorRect.width;

          if (this.horizontalViewportLock && this.anchorRect.right > this.viewportRect.right) {
            this.translateX = this.translateX - (this.anchorRect.right - this.viewportRect.right);
          }

          break;

        case "insetEnd":
          this.translateX = this.baseHorizontalOffset;

          if (this.horizontalViewportLock && this.anchorRect.left < this.viewportRect.left) {
            this.translateX = this.translateX - (this.anchorRect.left - this.viewportRect.left);
          }

          break;

        case "end":
          this.translateX = this.baseHorizontalOffset + this.anchorRect.width;

          if (this.horizontalViewportLock && this.anchorRect.right < this.viewportRect.left) {
            this.translateX = this.translateX - (this.anchorRect.right - this.viewportRect.left);
          }

          break;

        case "center":
          sizeDelta = (this.anchorRect.width - nextRegionWidth) / 2;
          this.translateX = this.baseHorizontalOffset + sizeDelta;

          if (this.horizontalViewportLock) {
            const regionLeft = this.anchorRect.left + sizeDelta;
            const regionRight = this.anchorRect.right - sizeDelta;

            if (regionLeft < this.viewportRect.left && !(regionRight > this.viewportRect.right)) {
              this.translateX = this.translateX - (regionLeft - this.viewportRect.left);
            } else if (regionRight > this.viewportRect.right && !(regionLeft < this.viewportRect.left)) {
              this.translateX = this.translateX - (regionRight - this.viewportRect.right);
            }
          }

          break;
      }

      this.horizontalPosition = desiredHorizontalPosition;
    };
    /**
     * Set vertical positioning state based on desired position
     */


    this.setVerticalPosition = (desiredVerticalPosition, nextPositionerDimension) => {
      if (desiredVerticalPosition === undefined || this.regionRect === undefined || this.anchorRect === undefined || this.viewportRect === undefined) {
        return;
      }

      let nextRegionHeight = 0;

      switch (this.verticalScaling) {
        case "anchor":
        case "fill":
          nextRegionHeight = this.verticalViewportLock ? this.viewportRect.height : nextPositionerDimension.height;
          this.regionHeight = `${nextRegionHeight}px`;
          break;

        case "content":
          nextRegionHeight = this.regionRect.height;
          this.regionHeight = "unset";
          break;
      }

      let sizeDelta = 0;

      switch (desiredVerticalPosition) {
        case "start":
          this.translateY = this.baseVerticalOffset - nextRegionHeight;

          if (this.verticalViewportLock && this.anchorRect.top > this.viewportRect.bottom) {
            this.translateY = this.translateY - (this.anchorRect.top - this.viewportRect.bottom);
          }

          break;

        case "insetStart":
          this.translateY = this.baseVerticalOffset - nextRegionHeight + this.anchorRect.height;

          if (this.verticalViewportLock && this.anchorRect.bottom > this.viewportRect.bottom) {
            this.translateY = this.translateY - (this.anchorRect.bottom - this.viewportRect.bottom);
          }

          break;

        case "insetEnd":
          this.translateY = this.baseVerticalOffset;

          if (this.verticalViewportLock && this.anchorRect.top < this.viewportRect.top) {
            this.translateY = this.translateY - (this.anchorRect.top - this.viewportRect.top);
          }

          break;

        case "end":
          this.translateY = this.baseVerticalOffset + this.anchorRect.height;

          if (this.verticalViewportLock && this.anchorRect.bottom < this.viewportRect.top) {
            this.translateY = this.translateY - (this.anchorRect.bottom - this.viewportRect.top);
          }

          break;

        case "center":
          sizeDelta = (this.anchorRect.height - nextRegionHeight) / 2;
          this.translateY = this.baseVerticalOffset + sizeDelta;

          if (this.verticalViewportLock) {
            const regionTop = this.anchorRect.top + sizeDelta;
            const regionBottom = this.anchorRect.bottom - sizeDelta;

            if (regionTop < this.viewportRect.top && !(regionBottom > this.viewportRect.bottom)) {
              this.translateY = this.translateY - (regionTop - this.viewportRect.top);
            } else if (regionBottom > this.viewportRect.bottom && !(regionTop < this.viewportRect.top)) {
              this.translateY = this.translateY - (regionBottom - this.viewportRect.bottom);
            }
          }

      }

      this.verticalPosition = desiredVerticalPosition;
    };
    /**
     *  Get available positions based on positioning mode
     */


    this.getPositioningOptions = inset => {
      if (inset) {
        return ["insetStart", "insetEnd"];
      }

      return ["start", "end"];
    };
    /**
     *  Get the space available for a particular relative position
     */


    this.getAvailableSpace = (positionOption, anchorStart, anchorEnd, anchorSpan, viewportStart, viewportEnd) => {
      const spaceStart = anchorStart - viewportStart;
      const spaceEnd = viewportEnd - (anchorStart + anchorSpan);

      switch (positionOption) {
        case "start":
          return spaceStart;

        case "insetStart":
          return spaceStart + anchorSpan;

        case "insetEnd":
          return spaceEnd + anchorSpan;

        case "end":
          return spaceEnd;

        case "center":
          return Math.min(spaceStart, spaceEnd) * 2 + anchorSpan;
      }
    };
    /**
     * Get region dimensions
     */


    this.getNextRegionDimension = (desiredHorizontalPosition, desiredVerticalPosition) => {
      const newRegionDimension = {
        height: this.regionRect !== undefined ? this.regionRect.height : 0,
        width: this.regionRect !== undefined ? this.regionRect.width : 0
      };

      if (desiredHorizontalPosition !== undefined && this.horizontalScaling === "fill") {
        newRegionDimension.width = this.getAvailableSpace(desiredHorizontalPosition, this.anchorRect !== undefined ? this.anchorRect.left : 0, this.anchorRect !== undefined ? this.anchorRect.right : 0, this.anchorRect !== undefined ? this.anchorRect.width : 0, this.viewportRect !== undefined ? this.viewportRect.left : 0, this.viewportRect !== undefined ? this.viewportRect.right : 0);
      } else if (this.horizontalScaling === "anchor") {
        newRegionDimension.width = this.anchorRect !== undefined ? this.anchorRect.width : 0;
      }

      if (desiredVerticalPosition !== undefined && this.verticalScaling === "fill") {
        newRegionDimension.height = this.getAvailableSpace(desiredVerticalPosition, this.anchorRect !== undefined ? this.anchorRect.top : 0, this.anchorRect !== undefined ? this.anchorRect.bottom : 0, this.anchorRect !== undefined ? this.anchorRect.height : 0, this.viewportRect !== undefined ? this.viewportRect.top : 0, this.viewportRect !== undefined ? this.viewportRect.bottom : 0);
      } else if (this.verticalScaling === "anchor") {
        newRegionDimension.height = this.anchorRect !== undefined ? this.anchorRect.height : 0;
      }

      return newRegionDimension;
    };
    /**
     * starts event listeners that can trigger auto updating
     */


    this.startAutoUpdateEventListeners = () => {
      window.addEventListener(eventResize, this.update, {
        passive: true
      });
      window.addEventListener(eventScroll, this.update, {
        passive: true,
        capture: true
      });

      if (this.resizeDetector !== null && this.viewportElement !== null) {
        this.resizeDetector.observe(this.viewportElement);
      }
    };
    /**
     * stops event listeners that can trigger auto updating
     */


    this.stopAutoUpdateEventListeners = () => {
      window.removeEventListener(eventResize, this.update);
      window.removeEventListener(eventScroll, this.update);

      if (this.resizeDetector !== null && this.viewportElement !== null) {
        this.resizeDetector.unobserve(this.viewportElement);
      }
    };
  }

  anchorChanged() {
    if (this.initialLayoutComplete) {
      this.anchorElement = this.getAnchor();
    }
  }

  viewportChanged() {
    if (this.initialLayoutComplete) {
      this.viewportElement = this.getViewport();
    }
  }

  horizontalPositioningModeChanged() {
    this.requestReset();
  }

  horizontalDefaultPositionChanged() {
    this.updateForAttributeChange();
  }

  horizontalViewportLockChanged() {
    this.updateForAttributeChange();
  }

  horizontalInsetChanged() {
    this.updateForAttributeChange();
  }

  horizontalThresholdChanged() {
    this.updateForAttributeChange();
  }

  horizontalScalingChanged() {
    this.updateForAttributeChange();
  }

  verticalPositioningModeChanged() {
    this.requestReset();
  }

  verticalDefaultPositionChanged() {
    this.updateForAttributeChange();
  }

  verticalViewportLockChanged() {
    this.updateForAttributeChange();
  }

  verticalInsetChanged() {
    this.updateForAttributeChange();
  }

  verticalThresholdChanged() {
    this.updateForAttributeChange();
  }

  verticalScalingChanged() {
    this.updateForAttributeChange();
  }

  fixedPlacementChanged() {
    if (this.$fastController.isConnected && this.initialLayoutComplete) {
      this.initialize();
    }
  }

  autoUpdateModeChanged(prevMode, newMode) {
    if (this.$fastController.isConnected && this.initialLayoutComplete) {
      if (prevMode === "auto") {
        this.stopAutoUpdateEventListeners();
      }

      if (newMode === "auto") {
        this.startAutoUpdateEventListeners();
      }
    }
  }

  anchorElementChanged() {
    this.requestReset();
  }

  viewportElementChanged() {
    if (this.$fastController.isConnected && this.initialLayoutComplete) {
      this.initialize();
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();

    if (this.autoUpdateMode === "auto") {
      this.startAutoUpdateEventListeners();
    }

    this.initialize();
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    super.disconnectedCallback();

    if (this.autoUpdateMode === "auto") {
      this.stopAutoUpdateEventListeners();
    }

    this.stopObservers();
    this.disconnectResizeDetector();
  }
  /**
   * @internal
   */


  adoptedCallback() {
    this.initialize();
  }
  /**
   * destroys the instance's resize observer
   */


  disconnectResizeDetector() {
    if (this.resizeDetector !== null) {
      this.resizeDetector.disconnect();
      this.resizeDetector = null;
    }
  }
  /**
   * initializes the instance's resize observer
   */


  initializeResizeDetector() {
    this.disconnectResizeDetector();
    this.resizeDetector = new window.ResizeObserver(this.handleResize);
  }
  /**
   * react to attribute changes that don't require a reset
   */


  updateForAttributeChange() {
    if (this.$fastController.isConnected && this.initialLayoutComplete) {
      this.forceUpdate = true;
      this.update();
    }
  }
  /**
   * fully initializes the component
   */


  initialize() {
    this.initializeResizeDetector();

    if (this.anchorElement === null) {
      this.anchorElement = this.getAnchor();
    }

    this.requestReset();
  }
  /**
   * Request a reset if there are currently no open requests
   */


  requestReset() {
    if (this.$fastController.isConnected && this.pendingReset === false) {
      this.setInitialState();
      DOM.queueUpdate(() => this.reset());
      this.pendingReset = true;
    }
  }
  /**
   * sets the starting configuration for component internal values
   */


  setInitialState() {
    this.initialLayoutComplete = false;
    this.regionVisible = false;
    this.translateX = 0;
    this.translateY = 0;
    this.baseHorizontalOffset = 0;
    this.baseVerticalOffset = 0;
    this.viewportRect = undefined;
    this.regionRect = undefined;
    this.anchorRect = undefined;
    this.verticalPosition = undefined;
    this.horizontalPosition = undefined;
    this.style.opacity = "0";
    this.style.pointerEvents = "none";
    this.forceUpdate = false;
    this.style.position = this.fixedPlacement ? "fixed" : "absolute";
    this.updatePositionClasses();
    this.updateRegionStyle();
  }

}
AnchoredRegion.intersectionService = new IntersectionService();

__decorate$1([attr], AnchoredRegion.prototype, "anchor", void 0);

__decorate$1([attr], AnchoredRegion.prototype, "viewport", void 0);

__decorate$1([attr({
  attribute: "horizontal-positioning-mode"
})], AnchoredRegion.prototype, "horizontalPositioningMode", void 0);

__decorate$1([attr({
  attribute: "horizontal-default-position"
})], AnchoredRegion.prototype, "horizontalDefaultPosition", void 0);

__decorate$1([attr({
  attribute: "horizontal-viewport-lock",
  mode: "boolean"
})], AnchoredRegion.prototype, "horizontalViewportLock", void 0);

__decorate$1([attr({
  attribute: "horizontal-inset",
  mode: "boolean"
})], AnchoredRegion.prototype, "horizontalInset", void 0);

__decorate$1([attr({
  attribute: "horizontal-threshold"
})], AnchoredRegion.prototype, "horizontalThreshold", void 0);

__decorate$1([attr({
  attribute: "horizontal-scaling"
})], AnchoredRegion.prototype, "horizontalScaling", void 0);

__decorate$1([attr({
  attribute: "vertical-positioning-mode"
})], AnchoredRegion.prototype, "verticalPositioningMode", void 0);

__decorate$1([attr({
  attribute: "vertical-default-position"
})], AnchoredRegion.prototype, "verticalDefaultPosition", void 0);

__decorate$1([attr({
  attribute: "vertical-viewport-lock",
  mode: "boolean"
})], AnchoredRegion.prototype, "verticalViewportLock", void 0);

__decorate$1([attr({
  attribute: "vertical-inset",
  mode: "boolean"
})], AnchoredRegion.prototype, "verticalInset", void 0);

__decorate$1([attr({
  attribute: "vertical-threshold"
})], AnchoredRegion.prototype, "verticalThreshold", void 0);

__decorate$1([attr({
  attribute: "vertical-scaling"
})], AnchoredRegion.prototype, "verticalScaling", void 0);

__decorate$1([attr({
  attribute: "fixed-placement",
  mode: "boolean"
})], AnchoredRegion.prototype, "fixedPlacement", void 0);

__decorate$1([attr({
  attribute: "auto-update-mode"
})], AnchoredRegion.prototype, "autoUpdateMode", void 0);

__decorate$1([observable], AnchoredRegion.prototype, "anchorElement", void 0);

__decorate$1([observable], AnchoredRegion.prototype, "viewportElement", void 0);

__decorate$1([observable], AnchoredRegion.prototype, "initialLayoutComplete", void 0);

/**
 * Building blocks - partial configs
 */

/**
 * A region that matches the size and position of the anchor horizontally
 */
const horizontalAnchorOverlay = {
  horizontalDefaultPosition: "center",
  horizontalPositioningMode: "locktodefault",
  horizontalInset: false,
  horizontalScaling: "anchor"
};
/**
 * Exported configs
 */

/**
 * A region that always places itself above the anchor, has
 * a width to match the anchor, and is sized vertically by content
 *
 * @public
 */

const FlyoutPosTop = Object.assign(Object.assign({}, horizontalAnchorOverlay), {
  verticalDefaultPosition: "top",
  verticalPositioningMode: "locktodefault",
  verticalInset: false,
  verticalScaling: "content"
});
/**
 * A region that always places itself below the anchor, has
 * a width to match the anchor, and is sized vertically by content
 *
 * @public
 */

const FlyoutPosBottom = Object.assign(Object.assign({}, horizontalAnchorOverlay), {
  verticalDefaultPosition: "bottom",
  verticalPositioningMode: "locktodefault",
  verticalInset: false,
  verticalScaling: "content"
});
/**
 * A region that places itself above or below the anchor
 * based on available space, has a width to match the anchor,
 * and is sized vertically by content
 *
 * @public
 */

const FlyoutPosTallest = Object.assign(Object.assign({}, horizontalAnchorOverlay), {
  verticalPositioningMode: "dynamic",
  verticalInset: false,
  verticalScaling: "content"
});
/**
 * A region that always places itself above the anchor, has
 * a width to match the anchor, and is sized vertically by available space
 *
 * @public
 */

const FlyoutPosTopFill = Object.assign(Object.assign({}, FlyoutPosTop), {
  verticalScaling: "fill"
});
/**
 * A region that always places itself below the anchor, has
 * a width to match the anchor, and is sized vertically by available space
 *
 * @public
 */

const FlyoutPosBottomFill = Object.assign(Object.assign({}, FlyoutPosBottom), {
  verticalScaling: "fill"
});
/**
 * A region that places itself above or below the anchor
 * based on available space, has a width to match the anchor,
 * and is sized vertically by available space
 *
 * @public
 */

const FlyoutPosTallestFill = Object.assign(Object.assign({}, FlyoutPosTallest), {
  verticalScaling: "fill"
});

/**
 * The template for {@link @microsoft/fast-foundation#Avatar} component.
 * @public
 */

const avatarTemplate = (context, definition) => html`<div class="backplate ${x => x.shape}" part="backplate" style="${x => x.fill ? `background-color: var(--avatar-fill-${x.fill});` : void 0}"><a class="link" part="link" href="${x => x.link ? x.link : void 0}" style="${x => x.color ? `color: var(--avatar-color-${x.color});` : void 0}"><slot name="media" part="media">${definition.media || ""}</slot><slot class="content" part="content"><slot></a></div><slot name="badge" part="badge"></slot>`;

/**
 * An Avatar Custom HTML Element
 *
 * @slot media - Used for media such as an image
 * @slot - The default slot for avatar text, commonly a name or initials
 * @slot badge - Used to provide a badge, such as a status badge
 * @csspart backplate - The wrapping container for the avatar
 * @csspart link - The avatar link
 * @csspart media - The media slot
 * @csspart content - The default slot
 *
 * @public
 */

class Avatar$1 extends FoundationElement {
  /**
   * Internal
   */
  connectedCallback() {
    super.connectedCallback();

    if (!this.shape) {
      this.shape = "circle";
    }
  }

}

__decorate$1([attr], Avatar$1.prototype, "fill", void 0);

__decorate$1([attr], Avatar$1.prototype, "color", void 0);

__decorate$1([attr], Avatar$1.prototype, "link", void 0);

__decorate$1([attr], Avatar$1.prototype, "shape", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#Badge} component.
 * @public
 */

const badgeTemplate = (context, definition) => html`<template class="${x => x.circular ? "circular" : ""}"><div class="control" part="control" style="${x => x.generateBadgeStyle()}"><slot></slot></div></template>`;

/**
 * A Badge Custom HTML Element.
 * @slot - The default slot for the badge
 * @csspart control - The element representing the badge, which wraps the default slot
 *
 * @public
 */

class Badge extends FoundationElement {
  constructor() {
    super(...arguments);

    this.generateBadgeStyle = () => {
      if (!this.fill && !this.color) {
        return;
      }

      const fill = `background-color: var(--badge-fill-${this.fill});`;
      const color = `color: var(--badge-color-${this.color});`;

      if (this.fill && !this.color) {
        return fill;
      } else if (this.color && !this.fill) {
        return color;
      } else {
        return `${color} ${fill}`;
      }
    };
  }

}

__decorate$1([attr({
  attribute: "fill"
})], Badge.prototype, "fill", void 0);

__decorate$1([attr({
  attribute: "color"
})], Badge.prototype, "color", void 0);

__decorate$1([attr({
  mode: "boolean"
})], Badge.prototype, "circular", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(BreadcrumbItem:class)} component.
 * @public
 */

const breadcrumbItemTemplate = (context, definition) => html`<div role="listitem" class="listitem" part="listitem">${when(x => x.href && x.href.length > 0, html` ${anchorTemplate(context, definition)} `)} ${when(x => !x.href, html` ${startSlotTemplate(context, definition)}<slot></slot>${endSlotTemplate(context, definition)} `)} ${when(x => x.separator, html`<span class="separator" part="separator" aria-hidden="true"><slot name="separator">${definition.separator || ""}</slot></span>`)}</div>`;

/**
 * A Breadcrumb Item Custom HTML Element.
 *
 * @public
 */

class BreadcrumbItem extends Anchor$1 {
  constructor() {
    super(...arguments);
    /**
     * @internal
     */

    this.separator = true;
  }

}

__decorate$1([observable], BreadcrumbItem.prototype, "separator", void 0);

applyMixins(BreadcrumbItem, StartEnd, DelegatesARIALink);

/**
 * The template for the {@link @microsoft/fast-foundation#Breadcrumb} component.
 * @public
 */

const breadcrumbTemplate = (context, definition) => html`<template role="navigation"><div role="list" class="list" part="list"><slot ${slotted({
  property: "slottedBreadcrumbItems",
  filter: elements()
})}></slot></div></template>`;

/**
 * A Breadcrumb Custom HTML Element.
 * @slot - The default slot for the breadcrumb items
 * @csspart list - The element wrapping the slotted items
 *
 * @public
 */

class Breadcrumb extends FoundationElement {
  slottedBreadcrumbItemsChanged() {
    if (this.$fastController.isConnected) {
      if (this.slottedBreadcrumbItems === undefined || this.slottedBreadcrumbItems.length === 0) {
        return;
      }

      const lastNode = this.slottedBreadcrumbItems[this.slottedBreadcrumbItems.length - 1];
      this.slottedBreadcrumbItems.forEach(item => {
        const itemIsLastNode = item === lastNode;
        this.setItemSeparator(item, itemIsLastNode);
        this.setAriaCurrent(item, itemIsLastNode);
      });
    }
  }

  setItemSeparator(item, isLastNode) {
    if (item instanceof BreadcrumbItem) {
      item.separator = !isLastNode;
    }
  }
  /**
   * Finds href on childnodes in the light DOM or shadow DOM.
   * We look in the shadow DOM because we insert an anchor when breadcrumb-item has an href.
   */


  findChildWithHref(node) {
    var _a, _b;

    if (node.childElementCount > 0) {
      return node.querySelector("a[href]");
    } else if ((_a = node.shadowRoot) === null || _a === void 0 ? void 0 : _a.childElementCount) {
      return (_b = node.shadowRoot) === null || _b === void 0 ? void 0 : _b.querySelector("a[href]");
    } else return null;
  }
  /**
   *  Sets ARIA Current for the current node
   * If child node with an anchor tag and with href is found then set aria-current to correct value for the child node,
   * otherwise apply aria-current to the host element, with an href
   */


  setAriaCurrent(item, isLastNode) {
    const childNodeWithHref = this.findChildWithHref(item);

    if (childNodeWithHref === null && item.hasAttribute("href") && item instanceof BreadcrumbItem) {
      isLastNode ? item.setAttribute("aria-current", "page") : item.removeAttribute("aria-current");
    } else if (childNodeWithHref !== null) {
      isLastNode ? childNodeWithHref.setAttribute("aria-current", "page") : childNodeWithHref.removeAttribute("aria-current");
    }
  }

}

__decorate$1([observable], Breadcrumb.prototype, "slottedBreadcrumbItems", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(Button:class)} component.
 * @public
 */

const buttonTemplate = (context, definition) => html`<button class="control" part="control" ?autofocus="${x => x.autofocus}" ?disabled="${x => x.disabled}" form="${x => x.formId}" formaction="${x => x.formaction}" formenctype="${x => x.formenctype}" formmethod="${x => x.formmethod}" formnovalidate="${x => x.formnovalidate}" formtarget="${x => x.formtarget}" name="${x => x.name}" type="${x => x.type}" value="${x => x.value}" aria-atomic="${x => x.ariaAtomic}" aria-busy="${x => x.ariaBusy}" aria-controls="${x => x.ariaControls}" aria-current="${x => x.ariaCurrent}" aria-describedby="${x => x.ariaDescribedby}" aria-details="${x => x.ariaDetails}" aria-disabled="${x => x.ariaDisabled}" aria-errormessage="${x => x.ariaErrormessage}" aria-expanded="${x => x.ariaExpanded}" aria-flowto="${x => x.ariaFlowto}" aria-haspopup="${x => x.ariaHaspopup}" aria-hidden="${x => x.ariaHidden}" aria-invalid="${x => x.ariaInvalid}" aria-keyshortcuts="${x => x.ariaKeyshortcuts}" aria-label="${x => x.ariaLabel}" aria-labelledby="${x => x.ariaLabelledby}" aria-live="${x => x.ariaLive}" aria-owns="${x => x.ariaOwns}" aria-pressed="${x => x.ariaPressed}" aria-relevant="${x => x.ariaRelevant}" aria-roledescription="${x => x.ariaRoledescription}" ${ref("control")}>${startSlotTemplate(context, definition)}<span class="content" part="content"><slot ${slotted("defaultSlottedContent")}></slot></span>${endSlotTemplate(context, definition)}</button>`;

const proxySlotName = "form-associated-proxy";
const ElementInternalsKey = "ElementInternals";
/**
 * @alpha
 */

const supportsElementInternals = ElementInternalsKey in window && "setFormValue" in window[ElementInternalsKey].prototype;
const InternalsMap = new WeakMap();
/**
 * Base function for providing Custom Element Form Association.
 *
 * @alpha
 */

function FormAssociated(BaseCtor) {
  const C = class extends BaseCtor {
    constructor(...args) {
      super(...args);
      /**
       * Track whether the value has been changed from the initial value
       */

      this.dirtyValue = false;
      /**
       * Sets the element's disabled state. A disabled element will not be included during form submission.
       *
       * @remarks
       * HTML Attribute: disabled
       */

      this.disabled = false;
      /**
       * These are events that are still fired by the proxy
       * element based on user / programmatic interaction.
       *
       * The proxy implementation should be transparent to
       * the app author, so block these events from emitting.
       */

      this.proxyEventsToBlock = ["change", "click"];
      this.proxyInitialized = false;
      this.required = false;
      this.initialValue = this.initialValue || "";

      if (!this.elementInternals) {
        // When elementInternals is not supported, formResetCallback is
        // bound to an event listener, so ensure the handler's `this`
        // context is correct.
        this.formResetCallback = this.formResetCallback.bind(this);
      }
    }
    /**
     * Must evaluate to true to enable elementInternals.
     * Feature detects API support and resolve respectively
     *
     * @internal
     */


    static get formAssociated() {
      return supportsElementInternals;
    }
    /**
     * Returns the validity state of the element
     *
     * @alpha
     */


    get validity() {
      return this.elementInternals ? this.elementInternals.validity : this.proxy.validity;
    }
    /**
     * Retrieve a reference to the associated form.
     * Returns null if not associated to any form.
     *
     * @alpha
     */


    get form() {
      return this.elementInternals ? this.elementInternals.form : this.proxy.form;
    }
    /**
     * Retrieve the localized validation message,
     * or custom validation message if set.
     *
     * @alpha
     */


    get validationMessage() {
      return this.elementInternals ? this.elementInternals.validationMessage : this.proxy.validationMessage;
    }
    /**
     * Whether the element will be validated when the
     * form is submitted
     */


    get willValidate() {
      return this.elementInternals ? this.elementInternals.willValidate : this.proxy.willValidate;
    }
    /**
     * A reference to all associated label elements
     */


    get labels() {
      if (this.elementInternals) {
        return Object.freeze(Array.from(this.elementInternals.labels));
      } else if (this.proxy instanceof HTMLElement && this.proxy.ownerDocument && this.id) {
        // Labels associated by wrapping the element: <label><custom-element></custom-element></label>
        const parentLabels = this.proxy.labels; // Labels associated using the `for` attribute

        const forLabels = Array.from(this.proxy.getRootNode().querySelectorAll(`[for='${this.id}']`));
        const labels = parentLabels ? forLabels.concat(Array.from(parentLabels)) : forLabels;
        return Object.freeze(labels);
      } else {
        return emptyArray;
      }
    }
    /**
     * Invoked when the `value` property changes
     * @param previous - the previous value
     * @param next - the new value
     *
     * @remarks
     * If elements extending `FormAssociated` implement a `valueChanged` method
     * They must be sure to invoke `super.valueChanged(previous, next)` to ensure
     * proper functioning of `FormAssociated`
     */


    valueChanged(previous, next) {
      this.dirtyValue = true;

      if (this.proxy instanceof HTMLElement) {
        this.proxy.value = this.value;
      }

      this.currentValue = this.value;
      this.setFormValue(this.value);
      this.validate();
    }

    currentValueChanged() {
      this.value = this.currentValue;
    }
    /**
     * Invoked when the `initialValue` property changes
     *
     * @param previous - the previous value
     * @param next - the new value
     *
     * @remarks
     * If elements extending `FormAssociated` implement a `initialValueChanged` method
     * They must be sure to invoke `super.initialValueChanged(previous, next)` to ensure
     * proper functioning of `FormAssociated`
     */


    initialValueChanged(previous, next) {
      // If the value is clean and the component is connected to the DOM
      // then set value equal to the attribute value.
      if (!this.dirtyValue) {
        this.value = this.initialValue;
        this.dirtyValue = false;
      }
    }
    /**
     * Invoked when the `disabled` property changes
     *
     * @param previous - the previous value
     * @param next - the new value
     *
     * @remarks
     * If elements extending `FormAssociated` implement a `disabledChanged` method
     * They must be sure to invoke `super.disabledChanged(previous, next)` to ensure
     * proper functioning of `FormAssociated`
     */


    disabledChanged(previous, next) {
      if (this.proxy instanceof HTMLElement) {
        this.proxy.disabled = this.disabled;
      }

      DOM.queueUpdate(() => this.classList.toggle("disabled", this.disabled));
    }
    /**
     * Invoked when the `name` property changes
     *
     * @param previous - the previous value
     * @param next - the new value
     *
     * @remarks
     * If elements extending `FormAssociated` implement a `nameChanged` method
     * They must be sure to invoke `super.nameChanged(previous, next)` to ensure
     * proper functioning of `FormAssociated`
     */


    nameChanged(previous, next) {
      if (this.proxy instanceof HTMLElement) {
        this.proxy.name = this.name;
      }
    }
    /**
     * Invoked when the `required` property changes
     *
     * @param previous - the previous value
     * @param next - the new value
     *
     * @remarks
     * If elements extending `FormAssociated` implement a `requiredChanged` method
     * They must be sure to invoke `super.requiredChanged(previous, next)` to ensure
     * proper functioning of `FormAssociated`
     */


    requiredChanged(prev, next) {
      if (this.proxy instanceof HTMLElement) {
        this.proxy.required = this.required;
      }

      DOM.queueUpdate(() => this.classList.toggle("required", this.required));
      this.validate();
    }
    /**
     * The element internals object. Will only exist
     * in browsers supporting the attachInternals API
     */


    get elementInternals() {
      if (!supportsElementInternals) {
        return null;
      }

      let internals = InternalsMap.get(this);

      if (!internals) {
        internals = this.attachInternals();
        InternalsMap.set(this, internals);
      }

      return internals;
    }
    /**
     * @internal
     */


    connectedCallback() {
      super.connectedCallback();
      this.addEventListener("keypress", this._keypressHandler);

      if (!this.value) {
        this.value = this.initialValue;
        this.dirtyValue = false;
      }

      if (!this.elementInternals) {
        this.attachProxy();

        if (this.form) {
          this.form.addEventListener("reset", this.formResetCallback);
        }
      }
    }
    /**
     * @internal
     */


    disconnectedCallback() {
      this.proxyEventsToBlock.forEach(name => this.proxy.removeEventListener(name, this.stopPropagation));

      if (!this.elementInternals && this.form) {
        this.form.removeEventListener("reset", this.formResetCallback);
      }
    }
    /**
     * Return the current validity of the element.
     */


    checkValidity() {
      return this.elementInternals ? this.elementInternals.checkValidity() : this.proxy.checkValidity();
    }
    /**
     * Return the current validity of the element.
     * If false, fires an invalid event at the element.
     */


    reportValidity() {
      return this.elementInternals ? this.elementInternals.reportValidity() : this.proxy.reportValidity();
    }
    /**
     * Set the validity of the control. In cases when the elementInternals object is not
     * available (and the proxy element is used to report validity), this function will
     * do nothing unless a message is provided, at which point the setCustomValidity method
     * of the proxy element will be invoked with the provided message.
     * @param flags - Validity flags
     * @param message - Optional message to supply
     * @param anchor - Optional element used by UA to display an interactive validation UI
     */


    setValidity(flags, message, anchor) {
      if (this.elementInternals) {
        this.elementInternals.setValidity(flags, message, anchor);
      } else if (typeof message === "string") {
        this.proxy.setCustomValidity(message);
      }
    }
    /**
     * Invoked when a connected component's form or fieldset has its disabled
     * state changed.
     * @param disabled - the disabled value of the form / fieldset
     */


    formDisabledCallback(disabled) {
      this.disabled = disabled;
    }

    formResetCallback() {
      this.value = this.initialValue;
      this.dirtyValue = false;
    }
    /**
     * Attach the proxy element to the DOM
     */


    attachProxy() {
      var _a;

      if (!this.proxyInitialized) {
        this.proxyInitialized = true;
        this.proxy.style.display = "none";
        this.proxyEventsToBlock.forEach(name => this.proxy.addEventListener(name, this.stopPropagation)); // These are typically mapped to the proxy during
        // property change callbacks, but during initialization
        // on the initial call of the callback, the proxy is
        // still undefined. We should find a better way to address this.

        this.proxy.disabled = this.disabled;
        this.proxy.required = this.required;

        if (typeof this.name === "string") {
          this.proxy.name = this.name;
        }

        if (typeof this.value === "string") {
          this.proxy.value = this.value;
        }

        this.proxy.setAttribute("slot", proxySlotName);
        this.proxySlot = document.createElement("slot");
        this.proxySlot.setAttribute("name", proxySlotName);
      }

      (_a = this.shadowRoot) === null || _a === void 0 ? void 0 : _a.appendChild(this.proxySlot);
      this.appendChild(this.proxy);
    }
    /**
     * Detach the proxy element from the DOM
     */


    detachProxy() {
      var _a;

      this.removeChild(this.proxy);
      (_a = this.shadowRoot) === null || _a === void 0 ? void 0 : _a.removeChild(this.proxySlot);
    }
    /**
     * Sets the validity of the custom element. By default this uses the proxy element to determine
     * validity, but this can be extended or replaced in implementation.
     */


    validate() {
      if (this.proxy instanceof HTMLElement) {
        this.setValidity(this.proxy.validity, this.proxy.validationMessage);
      }
    }
    /**
     * Associates the provided value (and optional state) with the parent form.
     * @param value - The value to set
     * @param state - The state object provided to during session restores and when autofilling.
     */


    setFormValue(value, state) {
      if (this.elementInternals) {
        this.elementInternals.setFormValue(value, state || value);
      }
    }

    _keypressHandler(e) {
      switch (e.key) {
        case keyEnter:
          if (this.form instanceof HTMLFormElement) {
            // Implicit submission
            const defaultButton = this.form.querySelector("[type=submit]");
            defaultButton === null || defaultButton === void 0 ? void 0 : defaultButton.click();
          }

          break;
      }
    }
    /**
     * Used to stop propagation of proxy element events
     * @param e - Event object
     */


    stopPropagation(e) {
      e.stopPropagation();
    }

  };
  attr({
    mode: "boolean"
  })(C.prototype, "disabled");
  attr({
    mode: "fromView",
    attribute: "value"
  })(C.prototype, "initialValue");
  attr({
    attribute: "current-value"
  })(C.prototype, "currentValue");
  attr(C.prototype, "name");
  attr({
    mode: "boolean"
  })(C.prototype, "required");
  observable(C.prototype, "value");
  return C;
}
/**
 * @alpha
 */

function CheckableFormAssociated(BaseCtor) {
  class C extends FormAssociated(BaseCtor) {}

  class D extends C {
    constructor(...args) {
      super(args);
      /**
       * Tracks whether the "checked" property has been changed.
       * This is necessary to provide consistent behavior with
       * normal input checkboxes
       */

      this.dirtyChecked = false;
      /**
       * Provides the default checkedness of the input element
       * Passed down to proxy
       *
       * @public
       * @remarks
       * HTML Attribute: checked
       */

      this.checkedAttribute = false;
      /**
       * The checked state of the control.
       *
       * @public
       */

      this.checked = false; // Re-initialize dirtyChecked because initialization of other values
      // causes it to become true

      this.dirtyChecked = false;
    }

    checkedAttributeChanged() {
      this.defaultChecked = this.checkedAttribute;
    }
    /**
     * @internal
     */


    defaultCheckedChanged() {
      if (!this.dirtyChecked) {
        // Setting this.checked will cause us to enter a dirty state,
        // but if we are clean when defaultChecked is changed, we want to stay
        // in a clean state, so reset this.dirtyChecked
        this.checked = this.defaultChecked;
        this.dirtyChecked = false;
      }
    }

    checkedChanged(prev, next) {
      if (!this.dirtyChecked) {
        this.dirtyChecked = true;
      }

      this.currentChecked = this.checked;
      this.updateForm();

      if (this.proxy instanceof HTMLInputElement) {
        this.proxy.checked = this.checked;
      }

      if (prev !== undefined) {
        this.$emit("change");
      }

      this.validate();
    }

    currentCheckedChanged(prev, next) {
      this.checked = this.currentChecked;
    }

    updateForm() {
      const value = this.checked ? this.value : null;
      this.setFormValue(value, value);
    }

    connectedCallback() {
      super.connectedCallback();
      this.updateForm();
    }

    formResetCallback() {
      super.formResetCallback();
      this.checked = !!this.checkedAttribute;
      this.dirtyChecked = false;
    }

  }

  attr({
    attribute: "checked",
    mode: "boolean"
  })(D.prototype, "checkedAttribute");
  attr({
    attribute: "current-checked",
    converter: booleanConverter
  })(D.prototype, "currentChecked");
  observable(D.prototype, "defaultChecked");
  observable(D.prototype, "checked");
  return D;
}

class _Button extends FoundationElement {}
/**
 * A form-associated base class for the {@link @microsoft/fast-foundation#(Button:class)} component.
 *
 * @internal
 */


class FormAssociatedButton extends FormAssociated(_Button) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("input");
  }

}

/**
 * A Button Custom HTML Element.
 * Based largely on the {@link https://developer.mozilla.org/en-US/docs/Web/HTML/Element/button | <button> element }.
 *
 * @slot start - Content which can be provided before the button content
 * @slot end - Content which can be provided after the button content
 * @slot - The default slot for button content
 * @csspart control - The button element
 * @csspart content - The element wrapping button content
 *
 * @public
 */

class Button$1 extends FormAssociatedButton {
  constructor() {
    super(...arguments);
    /**
     * Prevent events to propagate if disabled and has no slotted content wrapped in HTML elements
     * @internal
     */

    this.handleClick = e => {
      var _a;

      if (this.disabled && ((_a = this.defaultSlottedContent) === null || _a === void 0 ? void 0 : _a.length) <= 1) {
        e.stopPropagation();
      }
    };
    /**
     * Submits the parent form
     */


    this.handleSubmission = () => {
      if (!this.form) {
        return;
      }

      const attached = this.proxy.isConnected;

      if (!attached) {
        this.attachProxy();
      } // Browser support for requestSubmit is not comprehensive
      // so click the proxy if it isn't supported


      typeof this.form.requestSubmit === "function" ? this.form.requestSubmit(this.proxy) : this.proxy.click();

      if (!attached) {
        this.detachProxy();
      }
    };
    /**
     * Resets the parent form
     */


    this.handleFormReset = () => {
      var _a;

      (_a = this.form) === null || _a === void 0 ? void 0 : _a.reset();
    };
    /**
     * Overrides the focus call for where delegatesFocus is unsupported.
     * This check works for Chrome, Edge Chromium, FireFox, and Safari
     * Relevant PR on the Firefox browser: https://phabricator.services.mozilla.com/D123858
     */


    this.handleUnsupportedDelegatesFocus = () => {
      var _a; // Check to see if delegatesFocus is supported


      if (window.ShadowRoot && !window.ShadowRoot.prototype.hasOwnProperty("delegatesFocus") && ((_a = this.$fastController.definition.shadowOptions) === null || _a === void 0 ? void 0 : _a.delegatesFocus)) {
        this.focus = () => {
          this.control.focus();
        };
      }
    };
  }

  formactionChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.formAction = this.formaction;
    }
  }

  formenctypeChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.formEnctype = this.formenctype;
    }
  }

  formmethodChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.formMethod = this.formmethod;
    }
  }

  formnovalidateChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.formNoValidate = this.formnovalidate;
    }
  }

  formtargetChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.formTarget = this.formtarget;
    }
  }

  typeChanged(previous, next) {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.type = this.type;
    }

    next === "submit" && this.addEventListener("click", this.handleSubmission);
    previous === "submit" && this.removeEventListener("click", this.handleSubmission);
    next === "reset" && this.addEventListener("click", this.handleFormReset);
    previous === "reset" && this.removeEventListener("click", this.handleFormReset);
  }
  /**
   * @internal
   */


  connectedCallback() {
    var _a;

    super.connectedCallback();
    this.proxy.setAttribute("type", this.type);
    this.handleUnsupportedDelegatesFocus();
    const elements = Array.from((_a = this.control) === null || _a === void 0 ? void 0 : _a.children);

    if (elements) {
      elements.forEach(span => {
        span.addEventListener("click", this.handleClick);
      });
    }
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    var _a;

    super.disconnectedCallback();
    const elements = Array.from((_a = this.control) === null || _a === void 0 ? void 0 : _a.children);

    if (elements) {
      elements.forEach(span => {
        span.removeEventListener("click", this.handleClick);
      });
    }
  }

}

__decorate$1([attr({
  mode: "boolean"
})], Button$1.prototype, "autofocus", void 0);

__decorate$1([attr({
  attribute: "form"
})], Button$1.prototype, "formId", void 0);

__decorate$1([attr], Button$1.prototype, "formaction", void 0);

__decorate$1([attr], Button$1.prototype, "formenctype", void 0);

__decorate$1([attr], Button$1.prototype, "formmethod", void 0);

__decorate$1([attr({
  mode: "boolean"
})], Button$1.prototype, "formnovalidate", void 0);

__decorate$1([attr], Button$1.prototype, "formtarget", void 0);

__decorate$1([attr], Button$1.prototype, "type", void 0);

__decorate$1([observable], Button$1.prototype, "defaultSlottedContent", void 0);
/**
 * Includes ARIA states and properties relating to the ARIA button role
 *
 * @public
 */


class DelegatesARIAButton {}

__decorate$1([attr({
  attribute: "aria-expanded"
})], DelegatesARIAButton.prototype, "ariaExpanded", void 0);

__decorate$1([attr({
  attribute: "aria-pressed"
})], DelegatesARIAButton.prototype, "ariaPressed", void 0);

applyMixins(DelegatesARIAButton, ARIAGlobalStatesAndProperties);
applyMixins(Button$1, StartEnd, DelegatesARIAButton);

/**
 * Date formatting utility
 * @public
 */
class DateFormatter {
  constructor(config) {
    /**
     * Formatting for the day
     * @public
     */
    this.dayFormat = "numeric";
    /**
     * Formatting for the weekday labels
     * @public
     */

    this.weekdayFormat = "long";
    /**
     * Formatting for the month
     * @public
     */

    this.monthFormat = "long";
    /**
     * Formatting for the year
     * @public
     */

    this.yearFormat = "numeric";
    /**
     * Date used for formatting
     */

    this.date = new Date();
    /**
     * Add properties on construction
     */

    if (config) {
      for (const key in config) {
        const value = config[key];

        if (key === "date") {
          this.date = this.getDateObject(value);
        } else {
          this[key] = value;
        }
      }
    }
  }
  /**
   * Helper function to make sure that the DateFormatter is working with an instance of Date
   * @param date - The date as an object, string or Date insance
   * @returns - A Date instance
   * @public
   */


  getDateObject(date) {
    if (typeof date === "string") {
      const dates = date.split(/[/-]/);

      if (dates.length < 3) {
        return new Date();
      }

      return new Date(parseInt(dates[2], 10), parseInt(dates[0], 10) - 1, parseInt(dates[1], 10));
    } else if ("day" in date && "month" in date && "year" in date) {
      const {
        day,
        month,
        year
      } = date;
      return new Date(year, month - 1, day);
    }

    return date;
  }
  /**
   *
   * @param date - a valide date as either a Date, string, objec or a DateFormatter
   * @param format - The formatting for the string
   * @param locale - locale data used for formatting
   * @returns A localized string of the date provided
   * @public
   */


  getDate(date = this.date, format = {
    weekday: this.weekdayFormat,
    month: this.monthFormat,
    day: this.dayFormat,
    year: this.yearFormat
  }, locale = this.locale) {
    const dateObj = this.getDateObject(date);
    const optionsWithTimeZone = Object.assign({
      timeZone: "utc"
    }, format);
    return new Intl.DateTimeFormat(locale, optionsWithTimeZone).format(dateObj);
  }
  /**
   *
   * @param day - Day to localize
   * @param format - The formatting for the day
   * @param locale - The locale data used for formatting
   * @returns - A localized number for the day
   * @public
   */


  getDay(day = this.date.getDate(), format = this.dayFormat, locale = this.locale) {
    return this.getDate({
      month: 1,
      day,
      year: 2020
    }, {
      day: format
    }, locale);
  }
  /**
   *
   * @param month - The month to localize
   * @param format - The formatting for the month
   * @param locale - The locale data used for formatting
   * @returns - A localized name of the month
   * @public
   */


  getMonth(month = this.date.getMonth() + 1, format = this.monthFormat, locale = this.locale) {
    return this.getDate({
      month,
      day: 2,
      year: 2020
    }, {
      month: format
    }, locale);
  }
  /**
   *
   * @param year - The year to localize
   * @param format - The formatting for the year
   * @param locale - The locale data used for formatting
   * @returns - A localized string for the year
   * @public
   */


  getYear(year = this.date.getFullYear(), format = this.yearFormat, locale = this.locale) {
    return this.getDate({
      month: 2,
      day: 2,
      year
    }, {
      year: format
    }, locale);
  }
  /**
   *
   * @param weekday - The number of the weekday, defaults to Sunday
   * @param format - The formatting for the weekday label
   * @param locale - The locale data used for formatting
   * @returns - A formatted weekday label
   * @public
   */


  getWeekday(weekday = 0, format = this.weekdayFormat, locale = this.locale) {
    const date = `1-${weekday + 1}-2017`;
    return this.getDate(date, {
      weekday: format
    }, locale);
  }
  /**
   *
   * @param format - The formatting for the weekdays
   * @param locale - The locale data used for formatting
   * @returns - An array of the weekday labels
   * @public
   */


  getWeekdays(format = this.weekdayFormat, locale = this.locale) {
    return Array(7).fill(null).map((_, day) => this.getWeekday(day, format, locale));
  }

}

/**
 * Calendar component
 *
 * @slot - The default slot for calendar content
 * @fires dateselected - Fires a custom 'dateselected' event when Enter is invoked via keyboard on a date
 *
 * @public
 */

class Calendar extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * date formatter utitlity for getting localized strings
     * @public
     */

    this.dateFormatter = new DateFormatter();
    /**
     * Readonly attribute for turning off data-grid
     * @public
     */

    this.readonly = false;
    /**
     * String repesentation of the full locale including market, calendar type and numbering system
     * @public
     */

    this.locale = "en-US";
    /**
     * Month to display
     * @public
     */

    this.month = new Date().getMonth() + 1;
    /**
     * Year of the month to display
     * @public
     */

    this.year = new Date().getFullYear();
    /**
     * Format style for the day
     * @public
     */

    this.dayFormat = "numeric";
    /**
     * Format style for the week day labels
     * @public
     */

    this.weekdayFormat = "short";
    /**
     * Format style for the month label
     * @public
     */

    this.monthFormat = "long";
    /**
     * Format style for the year used in the title
     * @public
     */

    this.yearFormat = "numeric";
    /**
     * Minimum number of weeks to show for the month
     * This can be used to normalize the calendar view
     *  when changing or across multiple calendars
     * @public
     */

    this.minWeeks = 0;
    /**
     * A list of dates that should be shown as disabled
     * @public
     */

    this.disabledDates = "";
    /**
     * A list of dates that should be shown as highlighted
     * @public
     */

    this.selectedDates = "";
    /**
     * The number of miliseconds in a day
     * @internal
     */

    this.oneDayInMs = 86400000;
  }

  localeChanged() {
    this.dateFormatter.locale = this.locale;
  }

  dayFormatChanged() {
    this.dateFormatter.dayFormat = this.dayFormat;
  }

  weekdayFormatChanged() {
    this.dateFormatter.weekdayFormat = this.weekdayFormat;
  }

  monthFormatChanged() {
    this.dateFormatter.monthFormat = this.monthFormat;
  }

  yearFormatChanged() {
    this.dateFormatter.yearFormat = this.yearFormat;
  }
  /**
   * Gets data needed to render about a calendar month as well as the previous and next months
   * @param year - year of the calendar
   * @param month - month of the calendar
   * @returns - an object with data about the current and 2 surrounding months
   * @public
   */


  getMonthInfo(month = this.month, year = this.year) {
    const getFirstDay = date => new Date(date.getFullYear(), date.getMonth(), 1).getDay();

    const getLength = date => {
      const nextMonth = new Date(date.getFullYear(), date.getMonth() + 1, 1);
      return new Date(nextMonth.getTime() - this.oneDayInMs).getDate();
    };

    const thisMonth = new Date(year, month - 1);
    const nextMonth = new Date(year, month);
    const previousMonth = new Date(year, month - 2);
    return {
      length: getLength(thisMonth),
      month,
      start: getFirstDay(thisMonth),
      year,
      previous: {
        length: getLength(previousMonth),
        month: previousMonth.getMonth() + 1,
        start: getFirstDay(previousMonth),
        year: previousMonth.getFullYear()
      },
      next: {
        length: getLength(nextMonth),
        month: nextMonth.getMonth() + 1,
        start: getFirstDay(nextMonth),
        year: nextMonth.getFullYear()
      }
    };
  }
  /**
   * A list of calendar days
   * @param info - an object containing the information needed to render a calendar month
   * @param minWeeks - minimum number of weeks to show
   * @returns a list of days in a calendar month
   * @public
   */


  getDays(info = this.getMonthInfo(), minWeeks = this.minWeeks) {
    minWeeks = minWeeks > 10 ? 10 : minWeeks;
    const {
      start,
      length,
      previous,
      next
    } = info;
    const days = [];
    let dayCount = 1 - start;

    while (dayCount < length + 1 || days.length < minWeeks || days[days.length - 1].length % 7 !== 0) {
      const {
        month,
        year
      } = dayCount < 1 ? previous : dayCount > length ? next : info;
      const day = dayCount < 1 ? previous.length + dayCount : dayCount > length ? dayCount - length : dayCount;
      const dateString = `${month}-${day}-${year}`;
      const disabled = this.dateInString(dateString, this.disabledDates);
      const selected = this.dateInString(dateString, this.selectedDates);
      const date = {
        day,
        month,
        year,
        disabled,
        selected
      };
      const target = days[days.length - 1];

      if (days.length === 0 || target.length % 7 === 0) {
        days.push([date]);
      } else {
        target.push(date);
      }

      dayCount++;
    }

    return days;
  }
  /**
   * A helper function that checks if a date exists in a list of dates
   * @param date - A date objec that includes the day, month and year
   * @param datesString - a comma separated list of dates
   * @returns - Returns true if it found the date in the list of dates
   * @public
   */


  dateInString(date, datesString) {
    const dates = datesString.split(",").map(str => str.trim());
    date = typeof date === "string" ? date : `${date.getMonth() + 1}-${date.getDate()}-${date.getFullYear()}`;
    return dates.some(d => d === date);
  }
  /**
   * Creates a class string for the day container
   * @param date - date of the calendar cell
   * @returns - string of class names
   * @public
   */


  getDayClassNames(date, todayString) {
    const {
      day,
      month,
      year,
      disabled,
      selected
    } = date;
    const today = todayString === `${month}-${day}-${year}`;
    const inactive = this.month !== month;
    return ["day", today && "today", inactive && "inactive", disabled && "disabled", selected && "selected"].filter(Boolean).join(" ");
  }
  /**
   * Returns a list of weekday labels
   * @returns An array of weekday text and full text if abbreviated
   * @public
   */


  getWeekdayText() {
    const weekdayText = this.dateFormatter.getWeekdays().map(text => ({
      text
    }));

    if (this.weekdayFormat !== "long") {
      const longText = this.dateFormatter.getWeekdays("long");
      weekdayText.forEach((weekday, index) => {
        weekday.abbr = longText[index];
      });
    }

    return weekdayText;
  }
  /**
   * Emits the "date-select" event with the day, month and year.
   * @param date - Date cell
   * @public
   */


  handleDateSelect(event, day) {
    event.preventDefault;
    this.$emit("dateselected", day);
  }
  /**
   * Handles keyboard events on a cell
   * @param event - Keyboard event
   * @param date - Date of the cell selected
   */


  handleKeydown(event, date) {
    if (event.key === keyEnter) {
      this.handleDateSelect(event, date);
    }

    return true;
  }

}

__decorate$1([attr({
  mode: "boolean"
})], Calendar.prototype, "readonly", void 0);

__decorate$1([attr], Calendar.prototype, "locale", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], Calendar.prototype, "month", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], Calendar.prototype, "year", void 0);

__decorate$1([attr({
  attribute: "day-format",
  mode: "fromView"
})], Calendar.prototype, "dayFormat", void 0);

__decorate$1([attr({
  attribute: "weekday-format",
  mode: "fromView"
})], Calendar.prototype, "weekdayFormat", void 0);

__decorate$1([attr({
  attribute: "month-format",
  mode: "fromView"
})], Calendar.prototype, "monthFormat", void 0);

__decorate$1([attr({
  attribute: "year-format",
  mode: "fromView"
})], Calendar.prototype, "yearFormat", void 0);

__decorate$1([attr({
  attribute: "min-weeks",
  converter: nullableNumberConverter
})], Calendar.prototype, "minWeeks", void 0);

__decorate$1([attr({
  attribute: "disabled-dates"
})], Calendar.prototype, "disabledDates", void 0);

__decorate$1([attr({
  attribute: "selected-dates"
})], Calendar.prototype, "selectedDates", void 0);

/**
 * Enumerates the data grid auto generated header options
 * default option generates a non-sticky header row
 *
 * @public
 */
const GenerateHeaderOptions = {
  none: "none",
  default: "default",
  sticky: "sticky"
};
/**
 * Enumerates possible data grid cell types.
 *
 * @public
 */

const DataGridCellTypes = {
  default: "default",
  columnHeader: "columnheader",
  rowHeader: "rowheader"
};
/**
 * Enumerates possible data grid row types
 *
 * @public
 */

const DataGridRowTypes = {
  default: "default",
  header: "header",
  stickyHeader: "sticky-header"
};

/**
 * A Data Grid Row Custom HTML Element.
 *
 * @fires row-focused - Fires a custom 'row-focused' event when focus is on an element (usually a cell or its contents) in the row
 * @slot - The default slot for custom cell elements
 * @public
 */

class DataGridRow extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * The type of row
     *
     * @public
     * @remarks
     * HTML Attribute: row-type
     */

    this.rowType = DataGridRowTypes.default;
    /**
     * The base data for this row
     *
     * @public
     */

    this.rowData = null;
    /**
     * The column definitions of the row
     *
     * @public
     */

    this.columnDefinitions = null;
    /**
     * Whether focus is on/in a cell within this row.
     *
     * @internal
     */

    this.isActiveRow = false;
    this.cellsRepeatBehavior = null;
    this.cellsPlaceholder = null;
    /**
     * @internal
     */

    this.focusColumnIndex = 0;
    this.refocusOnLoad = false;

    this.updateRowStyle = () => {
      this.style.gridTemplateColumns = this.gridTemplateColumns;
    };
  }

  gridTemplateColumnsChanged() {
    if (this.$fastController.isConnected) {
      this.updateRowStyle();
    }
  }

  rowTypeChanged() {
    if (this.$fastController.isConnected) {
      this.updateItemTemplate();
    }
  }

  rowDataChanged() {
    if (this.rowData !== null && this.isActiveRow) {
      this.refocusOnLoad = true;
      return;
    }
  }

  cellItemTemplateChanged() {
    this.updateItemTemplate();
  }

  headerCellItemTemplateChanged() {
    this.updateItemTemplate();
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback(); // note that row elements can be reused with a different data object
    // as the parent grid's repeat behavior reacts to changes in the data set.

    if (this.cellsRepeatBehavior === null) {
      this.cellsPlaceholder = document.createComment("");
      this.appendChild(this.cellsPlaceholder);
      this.updateItemTemplate();
      this.cellsRepeatBehavior = new RepeatDirective(x => x.columnDefinitions, x => x.activeCellItemTemplate, {
        positioning: true
      }).createBehavior(this.cellsPlaceholder);
      /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */

      this.$fastController.addBehaviors([this.cellsRepeatBehavior]);
    }

    this.addEventListener("cell-focused", this.handleCellFocus);
    this.addEventListener(eventFocusOut, this.handleFocusout);
    this.addEventListener(eventKeyDown, this.handleKeydown);
    this.updateRowStyle();

    if (this.refocusOnLoad) {
      // if focus was on the row when data changed try to refocus on same cell
      this.refocusOnLoad = false;

      if (this.cellElements.length > this.focusColumnIndex) {
        this.cellElements[this.focusColumnIndex].focus();
      }
    }
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    super.disconnectedCallback();
    this.removeEventListener("cell-focused", this.handleCellFocus);
    this.removeEventListener(eventFocusOut, this.handleFocusout);
    this.removeEventListener(eventKeyDown, this.handleKeydown);
  }

  handleFocusout(e) {
    if (!this.contains(e.target)) {
      this.isActiveRow = false;
      this.focusColumnIndex = 0;
    }
  }

  handleCellFocus(e) {
    this.isActiveRow = true;
    this.focusColumnIndex = this.cellElements.indexOf(e.target);
    this.$emit("row-focused", this);
  }

  handleKeydown(e) {
    if (e.defaultPrevented) {
      return;
    }

    let newFocusColumnIndex = 0;

    switch (e.key) {
      case keyArrowLeft:
        // focus left one cell
        newFocusColumnIndex = Math.max(0, this.focusColumnIndex - 1);
        this.cellElements[newFocusColumnIndex].focus();
        e.preventDefault();
        break;

      case keyArrowRight:
        // focus right one cell
        newFocusColumnIndex = Math.min(this.cellElements.length - 1, this.focusColumnIndex + 1);
        this.cellElements[newFocusColumnIndex].focus();
        e.preventDefault();
        break;

      case keyHome:
        if (!e.ctrlKey) {
          this.cellElements[0].focus();
          e.preventDefault();
        }

        break;

      case keyEnd:
        if (!e.ctrlKey) {
          // focus last cell of the row
          this.cellElements[this.cellElements.length - 1].focus();
          e.preventDefault();
        }

        break;
    }
  }

  updateItemTemplate() {
    this.activeCellItemTemplate = this.rowType === DataGridRowTypes.default && this.cellItemTemplate !== undefined ? this.cellItemTemplate : this.rowType === DataGridRowTypes.default && this.cellItemTemplate === undefined ? this.defaultCellItemTemplate : this.headerCellItemTemplate !== undefined ? this.headerCellItemTemplate : this.defaultHeaderCellItemTemplate;
  }

}

__decorate$1([attr({
  attribute: "grid-template-columns"
})], DataGridRow.prototype, "gridTemplateColumns", void 0);

__decorate$1([attr({
  attribute: "row-type"
})], DataGridRow.prototype, "rowType", void 0);

__decorate$1([observable], DataGridRow.prototype, "rowData", void 0);

__decorate$1([observable], DataGridRow.prototype, "columnDefinitions", void 0);

__decorate$1([observable], DataGridRow.prototype, "cellItemTemplate", void 0);

__decorate$1([observable], DataGridRow.prototype, "headerCellItemTemplate", void 0);

__decorate$1([observable], DataGridRow.prototype, "rowIndex", void 0);

__decorate$1([observable], DataGridRow.prototype, "isActiveRow", void 0);

__decorate$1([observable], DataGridRow.prototype, "activeCellItemTemplate", void 0);

__decorate$1([observable], DataGridRow.prototype, "defaultCellItemTemplate", void 0);

__decorate$1([observable], DataGridRow.prototype, "defaultHeaderCellItemTemplate", void 0);

__decorate$1([observable], DataGridRow.prototype, "cellElements", void 0);

function createRowItemTemplate(context) {
  const rowTag = context.tagFor(DataGridRow);
  return html`<${rowTag} :rowData="${x => x}" :cellItemTemplate="${(x, c) => c.parent.cellItemTemplate}" :headerCellItemTemplate="${(x, c) => c.parent.headerCellItemTemplate}"></${rowTag}>`;
}
/**
 * Generates a template for the {@link @microsoft/fast-foundation#DataGrid} component using
 * the provided prefix.
 *
 * @public
 */


const dataGridTemplate = (context, definition) => {
  const rowItemTemplate = createRowItemTemplate(context);
  const rowTag = context.tagFor(DataGridRow);
  return html`<template role="grid" tabindex="0" :rowElementTag="${() => rowTag}" :defaultRowItemTemplate="${rowItemTemplate}" ${children({
    property: "rowElements",
    filter: elements("[role=row]")
  })}><slot></slot></template>`;
};

/**
 * A Data Grid Custom HTML Element.
 *
 * @slot - The default slot for custom row elements
 * @public
 */

class DataGrid extends FoundationElement {
  constructor() {
    super();
    /**
     * When true the component will not add itself to the tab queue.
     * Default is false.
     *
     * @public
     * @remarks
     * HTML Attribute: no-tabbing
     */

    this.noTabbing = false;
    /**
     *  Whether the grid should automatically generate a header row and its type
     *
     * @public
     * @remarks
     * HTML Attribute: generate-header
     */

    this.generateHeader = GenerateHeaderOptions.default;
    /**
     * The data being displayed in the grid
     *
     * @public
     */

    this.rowsData = [];
    /**
     * The column definitions of the grid
     *
     * @public
     */

    this.columnDefinitions = null;
    /**
     * The index of the row that will receive focus the next time the
     * grid is focused. This value changes as focus moves to different
     * rows within the grid.  Changing this value when focus is already
     * within the grid moves focus to the specified row.
     *
     * @public
     */

    this.focusRowIndex = 0;
    /**
     * The index of the column that will receive focus the next time the
     * grid is focused. This value changes as focus moves to different rows
     * within the grid.  Changing this value when focus is already within
     * the grid moves focus to the specified column.
     *
     * @public
     */

    this.focusColumnIndex = 0;
    this.rowsPlaceholder = null;
    this.generatedHeader = null;
    this.isUpdatingFocus = false;
    this.pendingFocusUpdate = false;
    this.rowindexUpdateQueued = false;
    this.columnDefinitionsStale = true;
    this.generatedGridTemplateColumns = "";

    this.focusOnCell = (rowIndex, columnIndex, scrollIntoView) => {
      if (this.rowElements.length === 0) {
        this.focusRowIndex = 0;
        this.focusColumnIndex = 0;
        return;
      }

      const focusRowIndex = Math.max(0, Math.min(this.rowElements.length - 1, rowIndex));
      const focusRow = this.rowElements[focusRowIndex];
      const cells = focusRow.querySelectorAll('[role="cell"], [role="gridcell"], [role="columnheader"], [role="rowheader"]');
      const focusColumnIndex = Math.max(0, Math.min(cells.length - 1, columnIndex));
      const focusTarget = cells[focusColumnIndex];

      if (scrollIntoView && this.scrollHeight !== this.clientHeight && (focusRowIndex < this.focusRowIndex && this.scrollTop > 0 || focusRowIndex > this.focusRowIndex && this.scrollTop < this.scrollHeight - this.clientHeight)) {
        focusTarget.scrollIntoView({
          block: "center",
          inline: "center"
        });
      }

      focusTarget.focus();
    };

    this.onChildListChange = (mutations,
    /* eslint-disable-next-line @typescript-eslint/no-unused-vars */
    observer) => {
      if (mutations && mutations.length) {
        mutations.forEach(mutation => {
          mutation.addedNodes.forEach(newNode => {
            if (newNode.nodeType === 1 && newNode.getAttribute("role") === "row") {
              newNode.columnDefinitions = this.columnDefinitions;
            }
          });
        });
        this.queueRowIndexUpdate();
      }
    };

    this.queueRowIndexUpdate = () => {
      if (!this.rowindexUpdateQueued) {
        this.rowindexUpdateQueued = true;
        DOM.queueUpdate(this.updateRowIndexes);
      }
    };

    this.updateRowIndexes = () => {
      let newGridTemplateColumns = this.gridTemplateColumns;

      if (newGridTemplateColumns === undefined) {
        // try to generate columns based on manual rows
        if (this.generatedGridTemplateColumns === "" && this.rowElements.length > 0) {
          const firstRow = this.rowElements[0];
          this.generatedGridTemplateColumns = new Array(firstRow.cellElements.length).fill("1fr").join(" ");
        }

        newGridTemplateColumns = this.generatedGridTemplateColumns;
      }

      this.rowElements.forEach((element, index) => {
        const thisRow = element;
        thisRow.rowIndex = index;
        thisRow.gridTemplateColumns = newGridTemplateColumns;

        if (this.columnDefinitionsStale) {
          thisRow.columnDefinitions = this.columnDefinitions;
        }
      });
      this.rowindexUpdateQueued = false;
      this.columnDefinitionsStale = false;
    };
  }
  /**
   *  generates a gridTemplateColumns based on columndata array
   */


  static generateTemplateColumns(columnDefinitions) {
    let templateColumns = "";
    columnDefinitions.forEach(column => {
      templateColumns = `${templateColumns}${templateColumns === "" ? "" : " "}${"1fr"}`;
    });
    return templateColumns;
  }

  noTabbingChanged() {
    if (this.$fastController.isConnected) {
      if (this.noTabbing) {
        this.setAttribute("tabIndex", "-1");
      } else {
        this.setAttribute("tabIndex", this.contains(document.activeElement) || this === document.activeElement ? "-1" : "0");
      }
    }
  }

  generateHeaderChanged() {
    if (this.$fastController.isConnected) {
      this.toggleGeneratedHeader();
    }
  }

  gridTemplateColumnsChanged() {
    if (this.$fastController.isConnected) {
      this.updateRowIndexes();
    }
  }

  rowsDataChanged() {
    if (this.columnDefinitions === null && this.rowsData.length > 0) {
      this.columnDefinitions = DataGrid.generateColumns(this.rowsData[0]);
    }

    if (this.$fastController.isConnected) {
      this.toggleGeneratedHeader();
    }
  }

  columnDefinitionsChanged() {
    if (this.columnDefinitions === null) {
      this.generatedGridTemplateColumns = "";
      return;
    }

    this.generatedGridTemplateColumns = DataGrid.generateTemplateColumns(this.columnDefinitions);

    if (this.$fastController.isConnected) {
      this.columnDefinitionsStale = true;
      this.queueRowIndexUpdate();
    }
  }

  headerCellItemTemplateChanged() {
    if (this.$fastController.isConnected) {
      if (this.generatedHeader !== null) {
        this.generatedHeader.headerCellItemTemplate = this.headerCellItemTemplate;
      }
    }
  }

  focusRowIndexChanged() {
    if (this.$fastController.isConnected) {
      this.queueFocusUpdate();
    }
  }

  focusColumnIndexChanged() {
    if (this.$fastController.isConnected) {
      this.queueFocusUpdate();
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();

    if (this.rowItemTemplate === undefined) {
      this.rowItemTemplate = this.defaultRowItemTemplate;
    }

    this.rowsPlaceholder = document.createComment("");
    this.appendChild(this.rowsPlaceholder);
    this.toggleGeneratedHeader();
    this.rowsRepeatBehavior = new RepeatDirective(x => x.rowsData, x => x.rowItemTemplate, {
      positioning: true
    }).createBehavior(this.rowsPlaceholder);
    /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */

    this.$fastController.addBehaviors([this.rowsRepeatBehavior]);
    this.addEventListener("row-focused", this.handleRowFocus);
    this.addEventListener(eventFocus, this.handleFocus);
    this.addEventListener(eventKeyDown, this.handleKeydown);
    this.addEventListener(eventFocusOut, this.handleFocusOut);
    this.observer = new MutationObserver(this.onChildListChange); // only observe if nodes are added or removed

    this.observer.observe(this, {
      childList: true
    });

    if (this.noTabbing) {
      this.setAttribute("tabindex", "-1");
    }

    DOM.queueUpdate(this.queueRowIndexUpdate);
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    super.disconnectedCallback();
    this.removeEventListener("row-focused", this.handleRowFocus);
    this.removeEventListener(eventFocus, this.handleFocus);
    this.removeEventListener(eventKeyDown, this.handleKeydown);
    this.removeEventListener(eventFocusOut, this.handleFocusOut); // disconnect observer

    this.observer.disconnect();
    this.rowsPlaceholder = null;
    this.generatedHeader = null;
  }
  /**
   * @internal
   */


  handleRowFocus(e) {
    this.isUpdatingFocus = true;
    const focusRow = e.target;
    this.focusRowIndex = this.rowElements.indexOf(focusRow);
    this.focusColumnIndex = focusRow.focusColumnIndex;
    this.setAttribute("tabIndex", "-1");
    this.isUpdatingFocus = false;
  }
  /**
   * @internal
   */


  handleFocus(e) {
    this.focusOnCell(this.focusRowIndex, this.focusColumnIndex, true);
  }
  /**
   * @internal
   */


  handleFocusOut(e) {
    if (e.relatedTarget === null || !this.contains(e.relatedTarget)) {
      this.setAttribute("tabIndex", this.noTabbing ? "-1" : "0");
    }
  }
  /**
   * @internal
   */


  handleKeydown(e) {
    if (e.defaultPrevented) {
      return;
    }

    let newFocusRowIndex;
    const maxIndex = this.rowElements.length - 1;
    const currentGridBottom = this.offsetHeight + this.scrollTop;
    const lastRow = this.rowElements[maxIndex];

    switch (e.key) {
      case keyArrowUp:
        e.preventDefault(); // focus up one row

        this.focusOnCell(this.focusRowIndex - 1, this.focusColumnIndex, true);
        break;

      case keyArrowDown:
        e.preventDefault(); // focus down one row

        this.focusOnCell(this.focusRowIndex + 1, this.focusColumnIndex, true);
        break;

      case keyPageUp:
        e.preventDefault();

        if (this.rowElements.length === 0) {
          this.focusOnCell(0, 0, false);
          break;
        }

        if (this.focusRowIndex === 0) {
          this.focusOnCell(0, this.focusColumnIndex, false);
          return;
        }

        newFocusRowIndex = this.focusRowIndex - 1;

        for (newFocusRowIndex; newFocusRowIndex >= 0; newFocusRowIndex--) {
          const thisRow = this.rowElements[newFocusRowIndex];

          if (thisRow.offsetTop < this.scrollTop) {
            this.scrollTop = thisRow.offsetTop + thisRow.clientHeight - this.clientHeight;
            break;
          }
        }

        this.focusOnCell(newFocusRowIndex, this.focusColumnIndex, false);
        break;

      case keyPageDown:
        e.preventDefault();

        if (this.rowElements.length === 0) {
          this.focusOnCell(0, 0, false);
          break;
        } // focus down one "page"


        if (this.focusRowIndex >= maxIndex || lastRow.offsetTop + lastRow.offsetHeight <= currentGridBottom) {
          this.focusOnCell(maxIndex, this.focusColumnIndex, false);
          return;
        }

        newFocusRowIndex = this.focusRowIndex + 1;

        for (newFocusRowIndex; newFocusRowIndex <= maxIndex; newFocusRowIndex++) {
          const thisRow = this.rowElements[newFocusRowIndex];

          if (thisRow.offsetTop + thisRow.offsetHeight > currentGridBottom) {
            let stickyHeaderOffset = 0;

            if (this.generateHeader === GenerateHeaderOptions.sticky && this.generatedHeader !== null) {
              stickyHeaderOffset = this.generatedHeader.clientHeight;
            }

            this.scrollTop = thisRow.offsetTop - stickyHeaderOffset;
            break;
          }
        }

        this.focusOnCell(newFocusRowIndex, this.focusColumnIndex, false);
        break;

      case keyHome:
        if (e.ctrlKey) {
          e.preventDefault(); // focus first cell of first row

          this.focusOnCell(0, 0, true);
        }

        break;

      case keyEnd:
        if (e.ctrlKey && this.columnDefinitions !== null) {
          e.preventDefault(); // focus last cell of last row

          this.focusOnCell(this.rowElements.length - 1, this.columnDefinitions.length - 1, true);
        }

        break;
    }
  }

  queueFocusUpdate() {
    if (this.isUpdatingFocus && (this.contains(document.activeElement) || this === document.activeElement)) {
      return;
    }

    if (this.pendingFocusUpdate === false) {
      this.pendingFocusUpdate = true;
      DOM.queueUpdate(() => this.updateFocus());
    }
  }

  updateFocus() {
    this.pendingFocusUpdate = false;
    this.focusOnCell(this.focusRowIndex, this.focusColumnIndex, true);
  }

  toggleGeneratedHeader() {
    if (this.generatedHeader !== null) {
      this.removeChild(this.generatedHeader);
      this.generatedHeader = null;
    }

    if (this.generateHeader !== GenerateHeaderOptions.none && this.rowsData.length > 0) {
      const generatedHeaderElement = document.createElement(this.rowElementTag);
      this.generatedHeader = generatedHeaderElement;
      this.generatedHeader.columnDefinitions = this.columnDefinitions;
      this.generatedHeader.gridTemplateColumns = this.gridTemplateColumns;
      this.generatedHeader.rowType = this.generateHeader === GenerateHeaderOptions.sticky ? DataGridRowTypes.stickyHeader : DataGridRowTypes.header;

      if (this.firstChild !== null || this.rowsPlaceholder !== null) {
        this.insertBefore(generatedHeaderElement, this.firstChild !== null ? this.firstChild : this.rowsPlaceholder);
      }

      return;
    }
  }

}
/**
 *  generates a basic column definition by examining sample row data
 */

DataGrid.generateColumns = row => {
  return Object.getOwnPropertyNames(row).map((property, index) => {
    return {
      columnDataKey: property,
      gridColumn: `${index}`
    };
  });
};

__decorate$1([attr({
  attribute: "no-tabbing",
  mode: "boolean"
})], DataGrid.prototype, "noTabbing", void 0);

__decorate$1([attr({
  attribute: "generate-header"
})], DataGrid.prototype, "generateHeader", void 0);

__decorate$1([attr({
  attribute: "grid-template-columns"
})], DataGrid.prototype, "gridTemplateColumns", void 0);

__decorate$1([observable], DataGrid.prototype, "rowsData", void 0);

__decorate$1([observable], DataGrid.prototype, "columnDefinitions", void 0);

__decorate$1([observable], DataGrid.prototype, "rowItemTemplate", void 0);

__decorate$1([observable], DataGrid.prototype, "cellItemTemplate", void 0);

__decorate$1([observable], DataGrid.prototype, "headerCellItemTemplate", void 0);

__decorate$1([observable], DataGrid.prototype, "focusRowIndex", void 0);

__decorate$1([observable], DataGrid.prototype, "focusColumnIndex", void 0);

__decorate$1([observable], DataGrid.prototype, "defaultRowItemTemplate", void 0);

__decorate$1([observable], DataGrid.prototype, "rowElementTag", void 0);

__decorate$1([observable], DataGrid.prototype, "rowElements", void 0);

const defaultCellContentsTemplate = html`<template>${x => x.rowData === null || x.columnDefinition === null || x.columnDefinition.columnDataKey === null ? null : x.rowData[x.columnDefinition.columnDataKey]}</template>`;
const defaultHeaderCellContentsTemplate = html`<template>${x => x.columnDefinition === null ? null : x.columnDefinition.title === undefined ? x.columnDefinition.columnDataKey : x.columnDefinition.title}</template>`;
/**
 * A Data Grid Cell Custom HTML Element.
 *
 * @fires cell-focused - Fires a custom 'cell-focused' event when focus is on the cell or its contents
 * @slot - The default slot for cell contents.  The "cell contents template" renders here.
 * @public
 */

class DataGridCell extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * The type of cell
     *
     * @public
     * @remarks
     * HTML Attribute: cell-type
     */

    this.cellType = DataGridCellTypes.default;
    /**
     * The base data for the parent row
     *
     * @public
     */

    this.rowData = null;
    /**
     * The base data for the column
     *
     * @public
     */

    this.columnDefinition = null;
    this.isActiveCell = false;
    this.customCellView = null;

    this.updateCellStyle = () => {
      this.style.gridColumn = this.gridColumn;
    };
  }

  cellTypeChanged() {
    if (this.$fastController.isConnected) {
      this.updateCellView();
    }
  }

  gridColumnChanged() {
    if (this.$fastController.isConnected) {
      this.updateCellStyle();
    }
  }

  columnDefinitionChanged(oldValue, newValue) {
    if (this.$fastController.isConnected) {
      this.updateCellView();
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    var _a;

    super.connectedCallback();
    this.addEventListener(eventFocusIn, this.handleFocusin);
    this.addEventListener(eventFocusOut, this.handleFocusout);
    this.addEventListener(eventKeyDown, this.handleKeydown);
    this.style.gridColumn = `${((_a = this.columnDefinition) === null || _a === void 0 ? void 0 : _a.gridColumn) === undefined ? 0 : this.columnDefinition.gridColumn}`;
    this.updateCellView();
    this.updateCellStyle();
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    super.disconnectedCallback();
    this.removeEventListener(eventFocusIn, this.handleFocusin);
    this.removeEventListener(eventFocusOut, this.handleFocusout);
    this.removeEventListener(eventKeyDown, this.handleKeydown);
    this.disconnectCellView();
  }

  handleFocusin(e) {
    if (this.isActiveCell) {
      return;
    }

    this.isActiveCell = true;

    switch (this.cellType) {
      case DataGridCellTypes.columnHeader:
        if (this.columnDefinition !== null && this.columnDefinition.headerCellInternalFocusQueue !== true && typeof this.columnDefinition.headerCellFocusTargetCallback === "function") {
          // move focus to the focus target
          const focusTarget = this.columnDefinition.headerCellFocusTargetCallback(this);

          if (focusTarget !== null) {
            focusTarget.focus();
          }
        }

        break;

      default:
        if (this.columnDefinition !== null && this.columnDefinition.cellInternalFocusQueue !== true && typeof this.columnDefinition.cellFocusTargetCallback === "function") {
          // move focus to the focus target
          const focusTarget = this.columnDefinition.cellFocusTargetCallback(this);

          if (focusTarget !== null) {
            focusTarget.focus();
          }
        }

        break;
    }

    this.$emit("cell-focused", this);
  }

  handleFocusout(e) {
    if (this !== document.activeElement && !this.contains(document.activeElement)) {
      this.isActiveCell = false;
    }
  }

  handleKeydown(e) {
    if (e.defaultPrevented || this.columnDefinition === null || this.cellType === DataGridCellTypes.default && this.columnDefinition.cellInternalFocusQueue !== true || this.cellType === DataGridCellTypes.columnHeader && this.columnDefinition.headerCellInternalFocusQueue !== true) {
      return;
    }

    switch (e.key) {
      case keyEnter:
      case keyFunction2:
        if (this.contains(document.activeElement) && document.activeElement !== this) {
          return;
        }

        switch (this.cellType) {
          case DataGridCellTypes.columnHeader:
            if (this.columnDefinition.headerCellFocusTargetCallback !== undefined) {
              const focusTarget = this.columnDefinition.headerCellFocusTargetCallback(this);

              if (focusTarget !== null) {
                focusTarget.focus();
              }

              e.preventDefault();
            }

            break;

          default:
            if (this.columnDefinition.cellFocusTargetCallback !== undefined) {
              const focusTarget = this.columnDefinition.cellFocusTargetCallback(this);

              if (focusTarget !== null) {
                focusTarget.focus();
              }

              e.preventDefault();
            }

            break;
        }

        break;

      case keyEscape:
        if (this.contains(document.activeElement) && document.activeElement !== this) {
          this.focus();
          e.preventDefault();
        }

        break;
    }
  }

  updateCellView() {
    this.disconnectCellView();

    if (this.columnDefinition === null) {
      return;
    }

    switch (this.cellType) {
      case DataGridCellTypes.columnHeader:
        if (this.columnDefinition.headerCellTemplate !== undefined) {
          this.customCellView = this.columnDefinition.headerCellTemplate.render(this, this);
        } else {
          this.customCellView = defaultHeaderCellContentsTemplate.render(this, this);
        }

        break;

      case undefined:
      case DataGridCellTypes.rowHeader:
      case DataGridCellTypes.default:
        if (this.columnDefinition.cellTemplate !== undefined) {
          this.customCellView = this.columnDefinition.cellTemplate.render(this, this);
        } else {
          this.customCellView = defaultCellContentsTemplate.render(this, this);
        }

        break;
    }
  }

  disconnectCellView() {
    if (this.customCellView !== null) {
      this.customCellView.dispose();
      this.customCellView = null;
    }
  }

}

__decorate$1([attr({
  attribute: "cell-type"
})], DataGridCell.prototype, "cellType", void 0);

__decorate$1([attr({
  attribute: "grid-column"
})], DataGridCell.prototype, "gridColumn", void 0);

__decorate$1([observable], DataGridCell.prototype, "rowData", void 0);

__decorate$1([observable], DataGridCell.prototype, "columnDefinition", void 0);

function createCellItemTemplate(context) {
  const cellTag = context.tagFor(DataGridCell);
  return html`<${cellTag} cell-type="${x => x.isRowHeader ? "rowheader" : undefined}" grid-column="${(x, c) => c.index + 1}" :rowData="${(x, c) => c.parent.rowData}" :columnDefinition="${x => x}"></${cellTag}>`;
}

function createHeaderCellItemTemplate(context) {
  const cellTag = context.tagFor(DataGridCell);
  return html`<${cellTag} cell-type="columnheader" grid-column="${(x, c) => c.index + 1}" :columnDefinition="${x => x}"></${cellTag}>`;
}
/**
 * Generates a template for the {@link @microsoft/fast-foundation#DataGridRow} component using
 * the provided prefix.
 *
 * @public
 */


const dataGridRowTemplate = (context, definition) => {
  const cellItemTemplate = createCellItemTemplate(context);
  const headerCellItemTemplate = createHeaderCellItemTemplate(context);
  return html`<template role="row" class="${x => x.rowType !== "default" ? x.rowType : ""}" :defaultCellItemTemplate="${cellItemTemplate}" :defaultHeaderCellItemTemplate="${headerCellItemTemplate}" ${children({
    property: "cellElements",
    filter: elements('[role="cell"],[role="gridcell"],[role="columnheader"],[role="rowheader"]')
  })}><slot ${slotted("slottedCellElements")}></slot></template>`;
};

/**
 * Generates a template for the {@link @microsoft/fast-foundation#DataGridCell} component using
 * the provided prefix.
 * @public
 */

const dataGridCellTemplate = (context, definition) => {
  return html`<template tabindex="-1" role="${x => !x.cellType || x.cellType === "default" ? "gridcell" : x.cellType}" class=" ${x => x.cellType === "columnheader" ? "column-header" : x.cellType === "rowheader" ? "row-header" : ""} "><slot></slot></template>`;
};

/**
 * A basic Calendar title template that includes the month and year
 * @returns - A calendar title template
 * @public
 */

const CalendarTitleTemplate = html`<div class="title" part="title" aria-label="${x => x.dateFormatter.getDate(`${x.month}-2-${x.year}`, {
  month: "long",
  year: "numeric"
})}"><span part="month">${x => x.dateFormatter.getMonth(x.month)}</span><span part="year">${x => x.dateFormatter.getYear(x.year)}</span></div>`;
/**
 * Calendar weekday label template
 * @returns - The weekday labels template
 * @public
 */

const calendarWeekdayTemplate = context => {
  const cellTag = context.tagFor(DataGridCell);
  return html`<${cellTag} class="week-day" part="week-day" tabindex="-1" grid-column="${(x, c) => c.index + 1}" abbr="${x => x.abbr}">${x => x.text}</${cellTag}>`;
};
/**
 * A calendar day template
 * @param context - Element definition context for getting the cell tag for calendar-cell
 * @param todayString - A string representation for todays date
 * @returns - A calendar cell template for a given date
 * @public
 */

const calendarCellTemplate = (context, todayString) => {
  const cellTag = context.tagFor(DataGridCell);
  return html`<${cellTag} class="${(x, c) => c.parentContext.parent.getDayClassNames(x, todayString)}" part="day" tabindex="-1" role="gridcell" grid-column="${(x, c) => c.index + 1}" @click="${(x, c) => c.parentContext.parent.handleDateSelect(c.event, x)}" @keydown="${(x, c) => c.parentContext.parent.handleKeydown(c.event, x)}" aria-label="${(x, c) => c.parentContext.parent.dateFormatter.getDate(`${x.month}-${x.day}-${x.year}`, {
    month: "long",
    day: "numeric"
  })}"><div class="date" part="${x => todayString === `${x.month}-${x.day}-${x.year}` ? "today" : "date"}">${(x, c) => c.parentContext.parent.dateFormatter.getDay(x.day)}</div><slot name="${x => x.month}-${x => x.day}-${x => x.year}"></slot></${cellTag}>`;
};
/**
 *
 * @param context - Element definition context for getting the cell tag for calendar-cell
 * @param todayString - A string representation for todays date
 * @returns - A template for a week of days
 * @public
 */

const calendarRowTemplate = (context, todayString) => {
  const rowTag = context.tagFor(DataGridRow);
  return html`<${rowTag} class="week" part="week" role="row" role-type="default" grid-template-columns="1fr 1fr 1fr 1fr 1fr 1fr 1fr">${repeat(x => x, calendarCellTemplate(context, todayString), {
    positioning: true
  })}</${rowTag}>`;
};
/**
 * Interactive template using DataGrid
 * @param context - The templates context
 * @param todayString - string representation of todays date
 * @returns - interactive calendar template
 *
 * @internal
 */

const interactiveCalendarGridTemplate = (context, todayString) => {
  const gridTag = context.tagFor(DataGrid);
  const rowTag = context.tagFor(DataGridRow);
  return html`<${gridTag} class="days interact" part="days" generate-header="none"><${rowTag} class="week-days" part="week-days" role="row" row-type="header" grid-template-columns="1fr 1fr 1fr 1fr 1fr 1fr 1fr">${repeat(x => x.getWeekdayText(), calendarWeekdayTemplate(context), {
    positioning: true
  })}</${rowTag}>${repeat(x => x.getDays(), calendarRowTemplate(context, todayString))}</${gridTag}>`;
};
/**
 * Non-interactive calendar template used for a readonly calendar
 * @param todayString - string representation of todays date
 * @returns - non-interactive calendar template
 *
 * @internal
 */

const noninteractiveCalendarTemplate = todayString => {
  return html`<div class="days" part="days"><div class="week-days" part="week-days">${repeat(x => x.getWeekdayText(), html`<div class="week-day" part="week-day" abbr="${x => x.abbr}">${x => x.text}</div>`)}</div>${repeat(x => x.getDays(), html`<div class="week">${repeat(x => x, html`<div class="${(x, c) => c.parentContext.parent.getDayClassNames(x, todayString)}" part="day" aria-label="${(x, c) => c.parentContext.parent.dateFormatter.getDate(`${x.month}-${x.day}-${x.year}`, {
    month: "long",
    day: "numeric"
  })}"><div class="date" part="${x => todayString === `${x.month}-${x.day}-${x.year}` ? "today" : "date"}">${(x, c) => c.parentContext.parent.dateFormatter.getDay(x.day)}</div><slot name="${x => x.month}-${x => x.day}-${x => x.year}"></slot></div>`)}</div>`)}</div>`;
};
/**
 * The template for the {@link @microsoft/fast-foundation#(Calendar:class)} component.
 *
 * @param context - Element definition context for getting the cell tag for calendar-cell
 * @param definition - Foundation element definition
 * @returns - a template for a calendar month
 * @public
 */

const calendarTemplate = (context, definition) => {
  var _a;

  const today = new Date();
  const todayString = `${today.getMonth() + 1}-${today.getDate()}-${today.getFullYear()}`;
  return html`<template>${startTemplate} ${definition.title instanceof Function ? definition.title(context, definition) : (_a = definition.title) !== null && _a !== void 0 ? _a : ""}<slot></slot>${when(x => x.readonly === false, interactiveCalendarGridTemplate(context, todayString))} ${when(x => x.readonly === true, noninteractiveCalendarTemplate(todayString))} ${endTemplate}</template>`;
};

/**
 * The template for the {@link @microsoft/fast-foundation#Card} component.
 * @public
 */

const cardTemplate = (context, definition) => html`<slot></slot>`;

/**
 * An Card Custom HTML Element.
 *
 * @slot - The default slot for the card content
 *
 * @public
 */

class Card$1 extends FoundationElement {}

/**
 * The template for the {@link @microsoft/fast-foundation#(Checkbox:class)} component.
 * @public
 */

const checkboxTemplate = (context, definition) => html`<template role="checkbox" aria-checked="${x => x.checked}" aria-required="${x => x.required}" aria-disabled="${x => x.disabled}" aria-readonly="${x => x.readOnly}" tabindex="${x => x.disabled ? null : 0}" @keypress="${(x, c) => x.keypressHandler(c.event)}" @click="${(x, c) => x.clickHandler(c.event)}" class="${x => x.readOnly ? "readonly" : ""} ${x => x.checked ? "checked" : ""} ${x => x.indeterminate ? "indeterminate" : ""}"><div part="control" class="control"><slot name="checked-indicator">${definition.checkedIndicator || ""}</slot><slot name="indeterminate-indicator">${definition.indeterminateIndicator || ""}</slot></div><label part="label" class="${x => x.defaultSlottedNodes && x.defaultSlottedNodes.length ? "label" : "label label__hidden"}"><slot ${slotted("defaultSlottedNodes")}></slot></label></template>`;

class _Checkbox extends FoundationElement {}
/**
 * A form-associated base class for the {@link @microsoft/fast-foundation#(Checkbox:class)} component.
 *
 * @internal
 */


class FormAssociatedCheckbox extends CheckableFormAssociated(_Checkbox) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("input");
  }

}

/**
 * A Checkbox Custom HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#checkbox | ARIA checkbox }.
 *
 * @slot checked-indicator - The checked indicator
 * @slot indeterminate-indicator - The indeterminate indicator
 * @slot - The default slot for the label
 * @csspart control - The element representing the visual checkbox control
 * @csspart label - The label
 * @fires change - Emits a custom change event when the checked state changes
 *
 * @public
 */

class Checkbox extends FormAssociatedCheckbox {
  constructor() {
    super();
    /**
     * The element's value to be included in form submission when checked.
     * Default to "on" to reach parity with input[type="checkbox"]
     *
     * @internal
     */

    this.initialValue = "on";
    /**
     * The indeterminate state of the control
     */

    this.indeterminate = false;
    /**
     * @internal
     */

    this.keypressHandler = e => {
      switch (e.key) {
        case keySpace:
          this.checked = !this.checked;
          break;
      }
    };
    /**
     * @internal
     */


    this.clickHandler = e => {
      if (!this.disabled && !this.readOnly) {
        this.checked = !this.checked;
      }
    };

    this.proxy.setAttribute("type", "checkbox");
  }

  readOnlyChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.readOnly = this.readOnly;
    }
  }

}

__decorate$1([attr({
  attribute: "readonly",
  mode: "boolean"
})], Checkbox.prototype, "readOnly", void 0);

__decorate$1([observable], Checkbox.prototype, "defaultSlottedNodes", void 0);

__decorate$1([observable], Checkbox.prototype, "indeterminate", void 0);

/**
 * Determines if the element is a {@link (ListboxOption:class)}
 *
 * @param element - the element to test.
 * @public
 */

function isListboxOption(el) {
  return isHTMLElement(el) && (el.getAttribute("role") === "option" || el instanceof HTMLOptionElement);
}
/**
 * An Option Custom HTML Element.
 * Implements {@link https://www.w3.org/TR/wai-aria-1.1/#option | ARIA option }.
 *
 * @slot start - Content which can be provided before the listbox option content
 * @slot end - Content which can be provided after the listbox option content
 * @slot - The default slot for listbox option content
 * @csspart content - Wraps the listbox option content
 *
 * @public
 */

class ListboxOption extends FoundationElement {
  constructor(text, value, defaultSelected, selected) {
    super();
    /**
     * The defaultSelected state of the option.
     * @public
     */

    this.defaultSelected = false;
    /**
     * Tracks whether the "selected" property has been changed.
     * @internal
     */

    this.dirtySelected = false;
    /**
     * The checked state of the control.
     *
     * @public
     */

    this.selected = this.defaultSelected;
    /**
     * Track whether the value has been changed from the initial value
     */

    this.dirtyValue = false;

    if (text) {
      this.textContent = text;
    }

    if (value) {
      this.initialValue = value;
    }

    if (defaultSelected) {
      this.defaultSelected = defaultSelected;
    }

    if (selected) {
      this.selected = selected;
    }

    this.proxy = new Option(`${this.textContent}`, this.initialValue, this.defaultSelected, this.selected);
    this.proxy.disabled = this.disabled;
  }
  /**
   * Updates the ariaChecked property when the checked property changes.
   *
   * @param prev - the previous checked value
   * @param next - the current checked value
   *
   * @public
   */


  checkedChanged(prev, next) {
    if (typeof next === "boolean") {
      this.ariaChecked = next ? "true" : "false";
      return;
    }

    this.ariaChecked = null;
  }
  /**
   * Updates the proxy's text content when the default slot changes.
   * @param prev - the previous content value
   * @param next - the current content value
   *
   * @internal
   */


  contentChanged(prev, next) {
    if (this.proxy instanceof HTMLOptionElement) {
      this.proxy.textContent = this.textContent;
    }

    this.$emit("contentchange", null, {
      bubbles: true
    });
  }

  defaultSelectedChanged() {
    if (!this.dirtySelected) {
      this.selected = this.defaultSelected;

      if (this.proxy instanceof HTMLOptionElement) {
        this.proxy.selected = this.defaultSelected;
      }
    }
  }

  disabledChanged(prev, next) {
    this.ariaDisabled = this.disabled ? "true" : "false";

    if (this.proxy instanceof HTMLOptionElement) {
      this.proxy.disabled = this.disabled;
    }
  }

  selectedAttributeChanged() {
    this.defaultSelected = this.selectedAttribute;

    if (this.proxy instanceof HTMLOptionElement) {
      this.proxy.defaultSelected = this.defaultSelected;
    }
  }

  selectedChanged() {
    this.ariaSelected = this.selected ? "true" : "false";

    if (!this.dirtySelected) {
      this.dirtySelected = true;
    }

    if (this.proxy instanceof HTMLOptionElement) {
      this.proxy.selected = this.selected;
    }
  }

  initialValueChanged(previous, next) {
    // If the value is clean and the component is connected to the DOM
    // then set value equal to the attribute value.
    if (!this.dirtyValue) {
      this.value = this.initialValue;
      this.dirtyValue = false;
    }
  }

  get label() {
    var _a;

    return (_a = this.value) !== null && _a !== void 0 ? _a : this.text;
  }

  get text() {
    var _a, _b;

    return (_b = (_a = this.textContent) === null || _a === void 0 ? void 0 : _a.replace(/\s+/g, " ").trim()) !== null && _b !== void 0 ? _b : "";
  }

  set value(next) {
    const newValue = `${next !== null && next !== void 0 ? next : ""}`;
    this._value = newValue;
    this.dirtyValue = true;

    if (this.proxy instanceof HTMLOptionElement) {
      this.proxy.value = newValue;
    }

    Observable.notify(this, "value");
  }

  get value() {
    var _a;

    Observable.track(this, "value");
    return (_a = this._value) !== null && _a !== void 0 ? _a : this.text;
  }

  get form() {
    return this.proxy ? this.proxy.form : null;
  }

}

__decorate$1([observable], ListboxOption.prototype, "checked", void 0);

__decorate$1([observable], ListboxOption.prototype, "content", void 0);

__decorate$1([observable], ListboxOption.prototype, "defaultSelected", void 0);

__decorate$1([attr({
  mode: "boolean"
})], ListboxOption.prototype, "disabled", void 0);

__decorate$1([attr({
  attribute: "selected",
  mode: "boolean"
})], ListboxOption.prototype, "selectedAttribute", void 0);

__decorate$1([observable], ListboxOption.prototype, "selected", void 0);

__decorate$1([attr({
  attribute: "value",
  mode: "fromView"
})], ListboxOption.prototype, "initialValue", void 0);
/**
 * States and properties relating to the ARIA `option` role.
 *
 * @public
 */


class DelegatesARIAListboxOption {}

__decorate$1([observable], DelegatesARIAListboxOption.prototype, "ariaChecked", void 0);

__decorate$1([observable], DelegatesARIAListboxOption.prototype, "ariaPosInSet", void 0);

__decorate$1([observable], DelegatesARIAListboxOption.prototype, "ariaSelected", void 0);

__decorate$1([observable], DelegatesARIAListboxOption.prototype, "ariaSetSize", void 0);

applyMixins(DelegatesARIAListboxOption, ARIAGlobalStatesAndProperties);
applyMixins(ListboxOption, StartEnd, DelegatesARIAListboxOption);

/**
 * A Listbox Custom HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#listbox | ARIA listbox }.
 *
 * @slot - The default slot for the listbox options
 *
 * @public
 */

class Listbox$1 extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * The internal unfiltered list of selectable options.
     *
     * @internal
     */

    this._options = [];
    /**
     * The index of the selected option.
     *
     * @public
     */

    this.selectedIndex = -1;
    /**
     * A collection of the selected options.
     *
     * @public
     */

    this.selectedOptions = [];
    /**
     * A standard `click` event creates a `focus` event before firing, so a
     * `mousedown` event is used to skip that initial focus.
     *
     * @internal
     */

    this.shouldSkipFocus = false;
    /**
     * The current typeahead buffer string.
     *
     * @internal
     */

    this.typeaheadBuffer = "";
    /**
     * Flag for the typeahead timeout expiration.
     *
     * @internal
     */

    this.typeaheadExpired = true;
    /**
     * The timeout ID for the typeahead handler.
     *
     * @internal
     */

    this.typeaheadTimeout = -1;
  }
  /**
   * The first selected option.
   *
   * @internal
   */


  get firstSelectedOption() {
    var _a;

    return (_a = this.selectedOptions[0]) !== null && _a !== void 0 ? _a : null;
  }
  /**
   * Returns true if there is one or more selectable option.
   *
   * @internal
   */


  get hasSelectableOptions() {
    return this.options.length > 0 && !this.options.every(o => o.disabled);
  }
  /**
   * The number of options.
   *
   * @public
   */


  get length() {
    var _a, _b;

    return (_b = (_a = this.options) === null || _a === void 0 ? void 0 : _a.length) !== null && _b !== void 0 ? _b : 0;
  }
  /**
   * The list of options.
   *
   * @public
   */


  get options() {
    Observable.track(this, "options");
    return this._options;
  }

  set options(value) {
    this._options = value;
    Observable.notify(this, "options");
  }
  /**
   * Flag for the typeahead timeout expiration.
   *
   * @deprecated use `Listbox.typeaheadExpired`
   * @internal
   */


  get typeAheadExpired() {
    return this.typeaheadExpired;
  }

  set typeAheadExpired(value) {
    this.typeaheadExpired = value;
  }
  /**
   * Handle click events for listbox options.
   *
   * @internal
   */


  clickHandler(e) {
    const captured = e.target.closest(`option,[role=option]`);

    if (captured && !captured.disabled) {
      this.selectedIndex = this.options.indexOf(captured);
      return true;
    }
  }
  /**
   * Ensures that the provided option is focused and scrolled into view.
   *
   * @param optionToFocus - The option to focus
   * @internal
   */


  focusAndScrollOptionIntoView(optionToFocus = this.firstSelectedOption) {
    // To ensure that the browser handles both `focus()` and `scrollIntoView()`, the
    // timing here needs to guarantee that they happen on different frames. Since this
    // function is typically called from the `openChanged` observer, `DOM.queueUpdate`
    // causes the calls to be grouped into the same frame. To prevent this,
    // `requestAnimationFrame` is used instead of `DOM.queueUpdate`.
    if (this.contains(document.activeElement) && optionToFocus !== null) {
      optionToFocus.focus();
      requestAnimationFrame(() => {
        optionToFocus.scrollIntoView({
          block: "nearest"
        });
      });
    }
  }
  /**
   * Handles `focusin` actions for the component. When the component receives focus,
   * the list of selected options is refreshed and the first selected option is scrolled
   * into view.
   *
   * @internal
   */


  focusinHandler(e) {
    if (!this.shouldSkipFocus && e.target === e.currentTarget) {
      this.setSelectedOptions();
      this.focusAndScrollOptionIntoView();
    }

    this.shouldSkipFocus = false;
  }
  /**
   * Returns the options which match the current typeahead buffer.
   *
   * @internal
   */


  getTypeaheadMatches() {
    const pattern = this.typeaheadBuffer.replace(/[.*+\-?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`^${pattern}`, "gi");
    return this.options.filter(o => o.text.trim().match(re));
  }
  /**
   * Determines the index of the next option which is selectable, if any.
   *
   * @param prev - the previous selected index
   * @param next - the next index to select
   *
   * @internal
   */


  getSelectableIndex(prev = this.selectedIndex, next) {
    const direction = prev > next ? -1 : prev < next ? 1 : 0;
    const potentialDirection = prev + direction;
    let nextSelectableOption = null;

    switch (direction) {
      case -1:
        {
          nextSelectableOption = this.options.reduceRight((nextSelectableOption, thisOption, index) => !nextSelectableOption && !thisOption.disabled && index < potentialDirection ? thisOption : nextSelectableOption, nextSelectableOption);
          break;
        }

      case 1:
        {
          nextSelectableOption = this.options.reduce((nextSelectableOption, thisOption, index) => !nextSelectableOption && !thisOption.disabled && index > potentialDirection ? thisOption : nextSelectableOption, nextSelectableOption);
          break;
        }
    }

    return this.options.indexOf(nextSelectableOption);
  }
  /**
   * Handles external changes to child options.
   *
   * @param source - the source object
   * @param propertyName - the property
   *
   * @internal
   */


  handleChange(source, propertyName) {
    switch (propertyName) {
      case "selected":
        {
          if (Listbox$1.slottedOptionFilter(source)) {
            this.selectedIndex = this.options.indexOf(source);
          }

          this.setSelectedOptions();
          break;
        }
    }
  }
  /**
   * Moves focus to an option whose label matches characters typed by the user.
   * Consecutive keystrokes are batched into a buffer of search text used
   * to match against the set of options.  If `TYPE_AHEAD_TIMEOUT_MS` passes
   * between consecutive keystrokes, the search restarts.
   *
   * @param key - the key to be evaluated
   *
   * @internal
   */


  handleTypeAhead(key) {
    if (this.typeaheadTimeout) {
      window.clearTimeout(this.typeaheadTimeout);
    }

    this.typeaheadTimeout = window.setTimeout(() => this.typeaheadExpired = true, Listbox$1.TYPE_AHEAD_TIMEOUT_MS);

    if (key.length > 1) {
      return;
    }

    this.typeaheadBuffer = `${this.typeaheadExpired ? "" : this.typeaheadBuffer}${key}`;
  }
  /**
   * Handles `keydown` actions for listbox navigation and typeahead.
   *
   * @internal
   */


  keydownHandler(e) {
    if (this.disabled) {
      return true;
    }

    this.shouldSkipFocus = false;
    const key = e.key;

    switch (key) {
      // Select the first available option
      case keyHome:
        {
          if (!e.shiftKey) {
            e.preventDefault();
            this.selectFirstOption();
          }

          break;
        }
      // Select the next selectable option

      case keyArrowDown:
        {
          if (!e.shiftKey) {
            e.preventDefault();
            this.selectNextOption();
          }

          break;
        }
      // Select the previous selectable option

      case keyArrowUp:
        {
          if (!e.shiftKey) {
            e.preventDefault();
            this.selectPreviousOption();
          }

          break;
        }
      // Select the last available option

      case keyEnd:
        {
          e.preventDefault();
          this.selectLastOption();
          break;
        }

      case keyTab:
        {
          this.focusAndScrollOptionIntoView();
          return true;
        }

      case keyEnter:
      case keyEscape:
        {
          return true;
        }

      case keySpace:
        {
          if (this.typeaheadExpired) {
            return true;
          }
        }
      // Send key to Typeahead handler

      default:
        {
          if (key.length === 1) {
            this.handleTypeAhead(`${key}`);
          }

          return true;
        }
    }
  }
  /**
   * Prevents `focusin` events from firing before `click` events when the
   * element is unfocused.
   *
   * @internal
   */


  mousedownHandler(e) {
    this.shouldSkipFocus = !this.contains(document.activeElement);
    return true;
  }
  /**
   * Switches between single-selection and multi-selection mode.
   *
   * @param prev - the previous value of the `multiple` attribute
   * @param next - the next value of the `multiple` attribute
   *
   * @internal
   */


  multipleChanged(prev, next) {
    this.ariaMultiSelectable = next ? "true" : null;
  }
  /**
   * Updates the list of selected options when the `selectedIndex` changes.
   *
   * @param prev - the previous selected index value
   * @param next - the current selected index value
   *
   * @internal
   */


  selectedIndexChanged(prev, next) {
    var _a;

    if (!this.hasSelectableOptions) {
      this.selectedIndex = -1;
      return;
    }

    if (((_a = this.options[this.selectedIndex]) === null || _a === void 0 ? void 0 : _a.disabled) && typeof prev === "number") {
      const selectableIndex = this.getSelectableIndex(prev, next);
      const newNext = selectableIndex > -1 ? selectableIndex : prev;
      this.selectedIndex = newNext;

      if (next === newNext) {
        this.selectedIndexChanged(next, newNext);
      }

      return;
    }

    this.setSelectedOptions();
  }
  /**
   * Updates the selectedness of each option when the list of selected options changes.
   *
   * @param prev - the previous list of selected options
   * @param next - the current list of selected options
   *
   * @internal
   */


  selectedOptionsChanged(prev, next) {
    var _a;

    const filteredNext = next.filter(Listbox$1.slottedOptionFilter);
    (_a = this.options) === null || _a === void 0 ? void 0 : _a.forEach(o => {
      const notifier = Observable.getNotifier(o);
      notifier.unsubscribe(this, "selected");
      o.selected = filteredNext.includes(o);
      notifier.subscribe(this, "selected");
    });
  }
  /**
   * Moves focus to the first selectable option.
   *
   * @public
   */


  selectFirstOption() {
    var _a, _b;

    if (!this.disabled) {
      this.selectedIndex = (_b = (_a = this.options) === null || _a === void 0 ? void 0 : _a.findIndex(o => !o.disabled)) !== null && _b !== void 0 ? _b : -1;
    }
  }
  /**
   * Moves focus to the last selectable option.
   *
   * @internal
   */


  selectLastOption() {
    if (!this.disabled) {
      this.selectedIndex = findLastIndex(this.options, o => !o.disabled);
    }
  }
  /**
   * Moves focus to the next selectable option.
   *
   * @internal
   */


  selectNextOption() {
    if (!this.disabled && this.selectedIndex < this.options.length - 1) {
      this.selectedIndex += 1;
    }
  }
  /**
   * Moves focus to the previous selectable option.
   *
   * @internal
   */


  selectPreviousOption() {
    if (!this.disabled && this.selectedIndex > 0) {
      this.selectedIndex = this.selectedIndex - 1;
    }
  }
  /**
   * Updates the selected index to match the first selected option.
   *
   * @internal
   */


  setDefaultSelectedOption() {
    var _a, _b;

    this.selectedIndex = (_b = (_a = this.options) === null || _a === void 0 ? void 0 : _a.findIndex(el => el.defaultSelected)) !== null && _b !== void 0 ? _b : -1;
  }
  /**
   * Sets an option as selected and gives it focus.
   *
   * @public
   */


  setSelectedOptions() {
    var _a, _b, _c;

    if ((_a = this.options) === null || _a === void 0 ? void 0 : _a.length) {
      this.selectedOptions = [this.options[this.selectedIndex]];
      this.ariaActiveDescendant = (_c = (_b = this.firstSelectedOption) === null || _b === void 0 ? void 0 : _b.id) !== null && _c !== void 0 ? _c : "";
      this.focusAndScrollOptionIntoView();
    }
  }
  /**
   * Updates the list of options and resets the selected option when the slotted option content changes.
   *
   * @param prev - the previous list of slotted options
   * @param next - the current list of slotted options
   *
   * @internal
   */


  slottedOptionsChanged(prev, next) {
    this.options = next.reduce((options, item) => {
      if (isListboxOption(item)) {
        options.push(item);
      }

      return options;
    }, []);
    const setSize = `${this.options.length}`;
    this.options.forEach((option, index) => {
      if (!option.id) {
        option.id = uniqueId("option-");
      }

      option.ariaPosInSet = `${index + 1}`;
      option.ariaSetSize = setSize;
    });

    if (this.$fastController.isConnected) {
      this.setSelectedOptions();
      this.setDefaultSelectedOption();
    }
  }
  /**
   * Updates the filtered list of options when the typeahead buffer changes.
   *
   * @param prev - the previous typeahead buffer value
   * @param next - the current typeahead buffer value
   *
   * @internal
   */


  typeaheadBufferChanged(prev, next) {
    if (this.$fastController.isConnected) {
      const typeaheadMatches = this.getTypeaheadMatches();

      if (typeaheadMatches.length) {
        const selectedIndex = this.options.indexOf(typeaheadMatches[0]);

        if (selectedIndex > -1) {
          this.selectedIndex = selectedIndex;
        }
      }

      this.typeaheadExpired = false;
    }
  }

}
/**
 * A static filter to include only selectable options.
 *
 * @param n - element to filter
 * @public
 */

Listbox$1.slottedOptionFilter = n => isListboxOption(n) && !n.hidden;
/**
 * Typeahead timeout in milliseconds.
 *
 * @internal
 */


Listbox$1.TYPE_AHEAD_TIMEOUT_MS = 1000;

__decorate$1([attr({
  mode: "boolean"
})], Listbox$1.prototype, "disabled", void 0);

__decorate$1([observable], Listbox$1.prototype, "selectedIndex", void 0);

__decorate$1([observable], Listbox$1.prototype, "selectedOptions", void 0);

__decorate$1([observable], Listbox$1.prototype, "slottedOptions", void 0);

__decorate$1([observable], Listbox$1.prototype, "typeaheadBuffer", void 0);
/**
 * Includes ARIA states and properties relating to the ARIA listbox role
 *
 * @public
 */


class DelegatesARIAListbox {}

__decorate$1([observable], DelegatesARIAListbox.prototype, "ariaActiveDescendant", void 0);

__decorate$1([observable], DelegatesARIAListbox.prototype, "ariaDisabled", void 0);

__decorate$1([observable], DelegatesARIAListbox.prototype, "ariaExpanded", void 0);

__decorate$1([observable], DelegatesARIAListbox.prototype, "ariaMultiSelectable", void 0);

applyMixins(DelegatesARIAListbox, ARIAGlobalStatesAndProperties);
applyMixins(Listbox$1, DelegatesARIAListbox);

/**
 * Positioning directions for the listbox when a select is open.
 * @public
 */
const SelectPosition = {
  above: "above",
  below: "below"
};

class _Combobox extends Listbox$1 {}
/**
 * A form-associated base class for the {@link (Combobox:class)} component.
 *
 * @internal
 */


class FormAssociatedCombobox extends FormAssociated(_Combobox) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("input");
  }

}

/**
 * Autocomplete values for combobox.
 * @public
 */
const ComboboxAutocomplete = {
  inline: "inline",
  list: "list",
  both: "both",
  none: "none"
};

/**
 * A Combobox Custom HTML Element.
 * Implements the {@link https://w3c.github.io/aria-practices/#combobox | ARIA combobox }.
 *
 * @slot start - Content which can be provided before the input
 * @slot end - Content which can be provided after the input
 * @slot control - Used to replace the input element representing the combobox
 * @slot indicator - The visual indicator representing the expanded state
 * @slot - The default slot for the options
 * @csspart control - The wrapper element containing the input area, including start and end
 * @csspart selected-value - The input element representing the selected value
 * @csspart indicator - The element wrapping the indicator slot
 * @csspart listbox - The wrapper for the listbox slotted options
 * @fires change - Fires a custom 'change' event when the value updates
 *
 * @public
 */

class Combobox$1 extends FormAssociatedCombobox {
  constructor() {
    super(...arguments);
    /**
     * The internal value property.
     *
     * @internal
     */

    this._value = "";
    /**
     * The collection of currently filtered options.
     *
     * @public
     */

    this.filteredOptions = [];
    /**
     * The current filter value.
     *
     * @internal
     */

    this.filter = "";
    /**
     * The initial state of the position attribute.
     *
     * @internal
     */

    this.forcedPosition = false;
    /**
     * The unique id for the internal listbox element.
     *
     * @internal
     */

    this.listboxId = uniqueId("listbox-");
    /**
     * The max height for the listbox when opened.
     *
     * @internal
     */

    this.maxHeight = 0;
    /**
     * The open attribute.
     *
     * @public
     * @remarks
     * HTML Attribute: open
     */

    this.open = false;
  }
  /**
   * Reset the element to its first selectable option when its parent form is reset.
   *
   * @internal
   */


  formResetCallback() {
    super.formResetCallback();
    this.setDefaultSelectedOption();
    this.updateValue();
  }

  get isAutocompleteInline() {
    return this.autocomplete === ComboboxAutocomplete.inline || this.isAutocompleteBoth;
  }

  get isAutocompleteList() {
    return this.autocomplete === ComboboxAutocomplete.list || this.isAutocompleteBoth;
  }

  get isAutocompleteBoth() {
    return this.autocomplete === ComboboxAutocomplete.both;
  }
  /**
   * Sets focus and synchronize ARIA attributes when the open property changes.
   *
   * @param prev - the previous open value
   * @param next - the current open value
   *
   * @internal
   */


  openChanged() {
    if (this.open) {
      this.ariaControls = this.listboxId;
      this.ariaExpanded = "true";
      this.setPositioning();
      this.focusAndScrollOptionIntoView(); // focus is directed to the element when `open` is changed programmatically

      DOM.queueUpdate(() => this.focus());
      return;
    }

    this.ariaControls = "";
    this.ariaExpanded = "false";
  }
  /**
   * The list of options.
   *
   * @public
   * @remarks
   * Overrides `Listbox.options`.
   */


  get options() {
    Observable.track(this, "options");
    return this.filteredOptions.length ? this.filteredOptions : this._options;
  }

  set options(value) {
    this._options = value;
    Observable.notify(this, "options");
  }
  /**
   * Updates the placeholder on the proxy element.
   * @internal
   */


  placeholderChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.placeholder = this.placeholder;
    }
  }

  positionChanged(prev, next) {
    this.positionAttribute = next;
    this.setPositioning();
  }
  /**
   * The value property.
   *
   * @public
   */


  get value() {
    Observable.track(this, "value");
    return this._value;
  }

  set value(next) {
    var _a, _b, _c;

    const prev = `${this._value}`;

    if (this.$fastController.isConnected && this.options) {
      const selectedIndex = this.options.findIndex(el => el.text.toLowerCase() === next.toLowerCase());
      const prevSelectedValue = (_a = this.options[this.selectedIndex]) === null || _a === void 0 ? void 0 : _a.text;
      const nextSelectedValue = (_b = this.options[selectedIndex]) === null || _b === void 0 ? void 0 : _b.text;
      this.selectedIndex = prevSelectedValue !== nextSelectedValue ? selectedIndex : this.selectedIndex;
      next = ((_c = this.firstSelectedOption) === null || _c === void 0 ? void 0 : _c.text) || next;
    }

    if (prev !== next) {
      this._value = next;
      super.valueChanged(prev, next);
      Observable.notify(this, "value");
    }
  }
  /**
   * Handle opening and closing the listbox when the combobox is clicked.
   *
   * @param e - the mouse event
   * @internal
   */


  clickHandler(e) {
    if (this.disabled) {
      return;
    }

    if (this.open) {
      const captured = e.target.closest(`option,[role=option]`);

      if (!captured || captured.disabled) {
        return;
      }

      this.selectedOptions = [captured];
      this.control.value = captured.text;
      this.clearSelectionRange();
      this.updateValue(true);
    }

    this.open = !this.open;

    if (this.open) {
      this.control.focus();
    }

    return true;
  }

  connectedCallback() {
    super.connectedCallback();
    this.forcedPosition = !!this.positionAttribute;

    if (this.value) {
      this.initialValue = this.value;
    }
  }
  /**
   * Synchronize the `aria-disabled` property when the `disabled` property changes.
   *
   * @param prev - The previous disabled value
   * @param next - The next disabled value
   *
   * @internal
   */


  disabledChanged(prev, next) {
    if (super.disabledChanged) {
      super.disabledChanged(prev, next);
    }

    this.ariaDisabled = this.disabled ? "true" : "false";
  }
  /**
   * Filter available options by text value.
   *
   * @public
   */


  filterOptions() {
    if (!this.autocomplete || this.autocomplete === ComboboxAutocomplete.none) {
      this.filter = "";
    }

    const filter = this.filter.toLowerCase();
    this.filteredOptions = this._options.filter(o => o.text.toLowerCase().startsWith(this.filter.toLowerCase()));

    if (this.isAutocompleteList) {
      if (!this.filteredOptions.length && !filter) {
        this.filteredOptions = this._options;
      }

      this._options.forEach(o => {
        o.hidden = !this.filteredOptions.includes(o);
      });
    }
  }
  /**
   * Focus the control and scroll the first selected option into view.
   *
   * @internal
   * @remarks
   * Overrides: `Listbox.focusAndScrollOptionIntoView`
   */


  focusAndScrollOptionIntoView() {
    if (this.contains(document.activeElement)) {
      this.control.focus();

      if (this.firstSelectedOption) {
        requestAnimationFrame(() => {
          var _a;

          (_a = this.firstSelectedOption) === null || _a === void 0 ? void 0 : _a.scrollIntoView({
            block: "nearest"
          });
        });
      }
    }
  }
  /**
   * Handle focus state when the element or its children lose focus.
   *
   * @param e - The focus event
   * @internal
   */


  focusoutHandler(e) {
    this.updateValue();

    if (!this.open) {
      return true;
    }

    const focusTarget = e.relatedTarget;

    if (this.isSameNode(focusTarget)) {
      this.focus();
      return;
    }

    if (!this.options || !this.options.includes(focusTarget)) {
      this.open = false;
    }
  }
  /**
   * Handle content changes on the control input.
   *
   * @param e - the input event
   * @internal
   */


  inputHandler(e) {
    this.filter = this.control.value;
    this.filterOptions();

    if (e.inputType === "deleteContentBackward" || !this.filter.length) {
      return true;
    }

    if (this.isAutocompleteList && !this.open) {
      this.open = true;
    }

    if (this.isAutocompleteInline && this.filteredOptions.length) {
      this.selectedOptions = [this.filteredOptions[0]];
      this.selectedIndex = this.options.indexOf(this.firstSelectedOption);
      this.setInlineSelection();
    }

    return;
  }
  /**
   * Handle keydown actions for listbox navigation.
   *
   * @param e - the keyboard event
   * @internal
   */


  keydownHandler(e) {
    const key = e.key;

    if (e.ctrlKey || e.shiftKey) {
      return true;
    }

    switch (key) {
      case "Enter":
        {
          this.updateValue(true);

          if (this.isAutocompleteInline) {
            this.filter = this.value;
          }

          this.open = false;
          this.clearSelectionRange();
          break;
        }

      case "Escape":
        {
          if (!this.isAutocompleteInline) {
            this.selectedIndex = -1;
          }

          if (this.open) {
            this.open = false;
            break;
          }

          this.value = "";
          this.control.value = "";
          this.filter = "";
          this.filterOptions();
          break;
        }

      case "Tab":
        {
          this.updateValue();

          if (!this.open) {
            return true;
          }

          e.preventDefault();
          this.open = false;
          break;
        }

      case "ArrowUp":
      case "ArrowDown":
        {
          this.filterOptions();

          if (!this.open) {
            this.open = true;
            break;
          }

          if (this.filteredOptions.length > 0) {
            super.keydownHandler(e);
          }

          if (this.isAutocompleteInline) {
            this.updateValue();
            this.setInlineSelection();
          }

          break;
        }

      default:
        {
          return true;
        }
    }
  }
  /**
   * Handle keyup actions for value input and text field manipulations.
   *
   * @param e - the keyboard event
   * @internal
   */


  keyupHandler(e) {
    const key = e.key;

    switch (key) {
      case "ArrowLeft":
      case "ArrowRight":
      case "Backspace":
      case "Delete":
      case "Home":
      case "End":
        {
          this.filter = this.control.value;
          this.selectedIndex = -1;
          this.filterOptions();
          break;
        }
    }
  }
  /**
   * Ensure that the selectedIndex is within the current allowable filtered range.
   *
   * @param prev - the previous selected index value
   * @param next - the current selected index value
   *
   * @internal
   */


  selectedIndexChanged(prev, next) {
    if (this.$fastController.isConnected) {
      next = limit(-1, this.options.length - 1, next); // we only want to call the super method when the selectedIndex is in range

      if (next !== this.selectedIndex) {
        this.selectedIndex = next;
        return;
      }

      super.selectedIndexChanged(prev, next);
    }
  }
  /**
   * Move focus to the previous selectable option.
   *
   * @internal
   * @remarks
   * Overrides `Listbox.selectPreviousOption`
   */


  selectPreviousOption() {
    if (!this.disabled && this.selectedIndex >= 0) {
      this.selectedIndex = this.selectedIndex - 1;
    }
  }
  /**
   * Set the default selected options at initialization or reset.
   *
   * @internal
   * @remarks
   * Overrides `Listbox.setDefaultSelectedOption`
   */


  setDefaultSelectedOption() {
    if (this.$fastController.isConnected && this.options) {
      const selectedIndex = this.options.findIndex(el => el.getAttribute("selected") !== null || el.selected);
      this.selectedIndex = selectedIndex;

      if (!this.dirtyValue && this.firstSelectedOption) {
        this.value = this.firstSelectedOption.text;
      }

      this.setSelectedOptions();
    }
  }
  /**
   * Focus and select the content of the control based on the first selected option.
   *
   * @param start - The index for the starting range
   * @internal
   */


  setInlineSelection() {
    if (this.firstSelectedOption) {
      this.control.value = this.firstSelectedOption.text;
      this.control.focus();
      this.control.setSelectionRange(this.filter.length, this.control.value.length, "backward");
    }
  }
  /**
   * Calculate and apply listbox positioning based on available viewport space.
   *
   * @param force - direction to force the listbox to display
   * @public
   */


  setPositioning() {
    const currentBox = this.getBoundingClientRect();
    const viewportHeight = window.innerHeight;
    const availableBottom = viewportHeight - currentBox.bottom;
    this.position = this.forcedPosition ? this.positionAttribute : currentBox.top > availableBottom ? SelectPosition.above : SelectPosition.below;
    this.positionAttribute = this.forcedPosition ? this.positionAttribute : this.position;
    this.maxHeight = this.position === SelectPosition.above ? ~~currentBox.top : ~~availableBottom;
  }
  /**
   * Ensure that the entire list of options is used when setting the selected property.
   *
   * @param prev - the previous list of selected options
   * @param next - the current list of selected options
   *
   * @internal
   * @remarks
   * Overrides: `Listbox.selectedOptionsChanged`
   */


  selectedOptionsChanged(prev, next) {
    if (this.$fastController.isConnected) {
      this._options.forEach(o => {
        o.selected = next.includes(o);
      });
    }
  }
  /**
   * Synchronize the form-associated proxy and update the value property of the element.
   *
   * @param prev - the previous collection of slotted option elements
   * @param next - the next collection of slotted option elements
   *
   * @internal
   */


  slottedOptionsChanged(prev, next) {
    super.slottedOptionsChanged(prev, next);
    this.updateValue();
  }
  /**
   * Sets the value and to match the first selected option.
   *
   * @param shouldEmit - if true, the change event will be emitted
   *
   * @internal
   */


  updateValue(shouldEmit) {
    var _a;

    if (this.$fastController.isConnected) {
      this.value = ((_a = this.firstSelectedOption) === null || _a === void 0 ? void 0 : _a.text) || this.control.value;
    }

    if (shouldEmit) {
      this.$emit("change");
    }
  }
  /**
   * @internal
   */


  clearSelectionRange() {
    const controlValueLength = this.control.value.length;
    this.control.setSelectionRange(controlValueLength, controlValueLength);
  }

}

__decorate$1([attr({
  attribute: "autocomplete",
  mode: "fromView"
})], Combobox$1.prototype, "autocomplete", void 0);

__decorate$1([observable], Combobox$1.prototype, "maxHeight", void 0);

__decorate$1([attr({
  attribute: "open",
  mode: "boolean"
})], Combobox$1.prototype, "open", void 0);

__decorate$1([attr], Combobox$1.prototype, "placeholder", void 0);

__decorate$1([attr({
  attribute: "position"
})], Combobox$1.prototype, "positionAttribute", void 0);

__decorate$1([observable], Combobox$1.prototype, "position", void 0);
/**
 * Includes ARIA states and properties relating to the ARIA combobox role.
 *
 * @public
 */


class DelegatesARIACombobox {}

__decorate$1([observable], DelegatesARIACombobox.prototype, "ariaAutoComplete", void 0);

__decorate$1([observable], DelegatesARIACombobox.prototype, "ariaControls", void 0);

applyMixins(DelegatesARIACombobox, DelegatesARIAListbox);
applyMixins(Combobox$1, StartEnd, DelegatesARIACombobox);

/**
 * The template for the {@link @microsoft/fast-foundation#(Combobox:class)} component.
 * @public
 */

const comboboxTemplate = (context, definition) => html`<template aria-disabled="${x => x.ariaDisabled}" autocomplete="${x => x.autocomplete}" class="${x => x.open ? "open" : ""} ${x => x.disabled ? "disabled" : ""} ${x => x.position}" ?open="${x => x.open}" tabindex="${x => !x.disabled ? "0" : null}" @click="${(x, c) => x.clickHandler(c.event)}" @focusout="${(x, c) => x.focusoutHandler(c.event)}" @keydown="${(x, c) => x.keydownHandler(c.event)}"><div class="control" part="control">${startSlotTemplate(context, definition)}<slot name="control"><input aria-activedescendant="${x => x.open ? x.ariaActiveDescendant : null}" aria-autocomplete="${x => x.ariaAutoComplete}" aria-controls="${x => x.ariaControls}" aria-disabled="${x => x.ariaDisabled}" aria-expanded="${x => x.ariaExpanded}" aria-haspopup="listbox" class="selected-value" part="selected-value" placeholder="${x => x.placeholder}" role="combobox" type="text" ?disabled="${x => x.disabled}" :value="${x => x.value}" @input="${(x, c) => x.inputHandler(c.event)}" @keyup="${(x, c) => x.keyupHandler(c.event)}" ${ref("control")} /><div class="indicator" part="indicator" aria-hidden="true"><slot name="indicator">${definition.indicator || ""}</slot></div></slot>${endSlotTemplate(context, definition)}</div><div class="listbox" id="${x => x.listboxId}" part="listbox" role="listbox" ?disabled="${x => x.disabled}" ?hidden="${x => !x.open}" ${ref("listbox")}><slot ${slotted({
  filter: Listbox$1.slottedOptionFilter,
  flatten: true,
  property: "slottedOptions"
})}></slot></div></template>`;

/**
 * Retrieves the "composed parent" element of a node, ignoring DOM tree boundaries.
 * When the parent of a node is a shadow-root, it will return the host
 * element of the shadow root. Otherwise it will return the parent node or null if
 * no parent node exists.
 * @param element - The element for which to retrieve the composed parent
 *
 * @public
 */
function composedParent(element) {
  const parentNode = element.parentElement;

  if (parentNode) {
    return parentNode;
  } else {
    const rootNode = element.getRootNode();

    if (rootNode.host instanceof HTMLElement) {
      // this is shadow-root
      return rootNode.host;
    }
  }

  return null;
}

/**
 * Determines if the reference element contains the test element in a "composed" DOM tree that
 * ignores shadow DOM boundaries.
 *
 * Returns true of the test element is a descendent of the reference, or exist in
 * a shadow DOM that is a logical descendent of the reference. Otherwise returns false.
 * @param reference - The element to test for containment against.
 * @param test - The element being tested for containment.
 *
 * @public
 */

function composedContains(reference, test) {
  let current = test;

  while (current !== null) {
    if (current === reference) {
      return true;
    }

    current = composedParent(current);
  }

  return false;
}

const defaultElement = document.createElement("div");

function isFastElement(element) {
  return element instanceof FASTElement;
}

class QueuedStyleSheetTarget {
  setProperty(name, value) {
    DOM.queueUpdate(() => this.target.setProperty(name, value));
  }

  removeProperty(name) {
    DOM.queueUpdate(() => this.target.removeProperty(name));
  }

}
/**
 * Handles setting properties for a FASTElement using Constructable Stylesheets
 */


class ConstructableStyleSheetTarget extends QueuedStyleSheetTarget {
  constructor(source) {
    super();
    const sheet = new CSSStyleSheet();
    this.target = sheet.cssRules[sheet.insertRule(":host{}")].style;
    source.$fastController.addStyles(ElementStyles.create([sheet]));
  }

}

class DocumentStyleSheetTarget extends QueuedStyleSheetTarget {
  constructor() {
    super();
    const sheet = new CSSStyleSheet();
    this.target = sheet.cssRules[sheet.insertRule(":root{}")].style;
    document.adoptedStyleSheets = [...document.adoptedStyleSheets, sheet];
  }

}

class HeadStyleElementStyleSheetTarget extends QueuedStyleSheetTarget {
  constructor() {
    super();
    this.style = document.createElement("style");
    document.head.appendChild(this.style);
    const {
      sheet
    } = this.style; // Because the HTMLStyleElement has been appended,
    // there shouldn't exist a case where `sheet` is null,
    // but if-check it just in case.

    if (sheet) {
      // https://github.com/jsdom/jsdom uses https://github.com/NV/CSSOM for it's CSSOM implementation,
      // which implements the DOM Level 2 spec for CSSStyleSheet where insertRule() requires an index argument.
      const index = sheet.insertRule(":root{}", sheet.cssRules.length);
      this.target = sheet.cssRules[index].style;
    }
  }

}
/**
 * Handles setting properties for a FASTElement using an HTMLStyleElement
 */


class StyleElementStyleSheetTarget {
  constructor(target) {
    this.store = new Map();
    this.target = null;
    const controller = target.$fastController;
    this.style = document.createElement("style");
    controller.addStyles(this.style);
    Observable.getNotifier(controller).subscribe(this, "isConnected");
    this.handleChange(controller, "isConnected");
  }

  targetChanged() {
    if (this.target !== null) {
      for (const [key, value] of this.store.entries()) {
        this.target.setProperty(key, value);
      }
    }
  }

  setProperty(name, value) {
    this.store.set(name, value);
    DOM.queueUpdate(() => {
      if (this.target !== null) {
        this.target.setProperty(name, value);
      }
    });
  }

  removeProperty(name) {
    this.store.delete(name);
    DOM.queueUpdate(() => {
      if (this.target !== null) {
        this.target.removeProperty(name);
      }
    });
  }

  handleChange(source, key) {
    // HTMLStyleElement.sheet is null if the element isn't connected to the DOM,
    // so this method reacts to changes in DOM connection for the element hosting
    // the HTMLStyleElement.
    //
    // All rules applied via the CSSOM also get cleared when the element disconnects,
    // so we need to add a new rule each time and populate it with the stored properties
    const {
      sheet
    } = this.style;

    if (sheet) {
      // Safari will throw if we try to use the return result of insertRule()
      // to index the rule inline, so store as a const prior to indexing.
      // https://github.com/jsdom/jsdom uses https://github.com/NV/CSSOM for it's CSSOM implementation,
      // which implements the DOM Level 2 spec for CSSStyleSheet where insertRule() requires an index argument.
      const index = sheet.insertRule(":host{}", sheet.cssRules.length);
      this.target = sheet.cssRules[index].style;
    } else {
      this.target = null;
    }
  }

}

__decorate$1([observable], StyleElementStyleSheetTarget.prototype, "target", void 0);
/**
 * Handles setting properties for a normal HTMLElement
 */


class ElementStyleSheetTarget {
  constructor(source) {
    this.target = source.style;
  }

  setProperty(name, value) {
    DOM.queueUpdate(() => this.target.setProperty(name, value));
  }

  removeProperty(name) {
    DOM.queueUpdate(() => this.target.removeProperty(name));
  }

}
/**
 * Controls emission for default values. This control is capable
 * of emitting to multiple {@link PropertyTarget | PropertyTargets},
 * and only emits if it has at least one root.
 *
 * @internal
 */


class RootStyleSheetTarget {
  setProperty(name, value) {
    RootStyleSheetTarget.properties[name] = value;

    for (const target of RootStyleSheetTarget.roots.values()) {
      PropertyTargetManager.getOrCreate(RootStyleSheetTarget.normalizeRoot(target)).setProperty(name, value);
    }
  }

  removeProperty(name) {
    delete RootStyleSheetTarget.properties[name];

    for (const target of RootStyleSheetTarget.roots.values()) {
      PropertyTargetManager.getOrCreate(RootStyleSheetTarget.normalizeRoot(target)).removeProperty(name);
    }
  }

  static registerRoot(root) {
    const {
      roots
    } = RootStyleSheetTarget;

    if (!roots.has(root)) {
      roots.add(root);
      const target = PropertyTargetManager.getOrCreate(this.normalizeRoot(root));

      for (const key in RootStyleSheetTarget.properties) {
        target.setProperty(key, RootStyleSheetTarget.properties[key]);
      }
    }
  }

  static unregisterRoot(root) {
    const {
      roots
    } = RootStyleSheetTarget;

    if (roots.has(root)) {
      roots.delete(root);
      const target = PropertyTargetManager.getOrCreate(RootStyleSheetTarget.normalizeRoot(root));

      for (const key in RootStyleSheetTarget.properties) {
        target.removeProperty(key);
      }
    }
  }
  /**
   * Returns the document when provided the default element,
   * otherwise is a no-op
   * @param root - the root to normalize
   */


  static normalizeRoot(root) {
    return root === defaultElement ? document : root;
  }

}
RootStyleSheetTarget.roots = new Set();
RootStyleSheetTarget.properties = {}; // Caches PropertyTarget instances

const propertyTargetCache = new WeakMap(); // Use Constructable StyleSheets for FAST elements when supported, otherwise use
// HTMLStyleElement instances

const propertyTargetCtor = DOM.supportsAdoptedStyleSheets ? ConstructableStyleSheetTarget : StyleElementStyleSheetTarget;
/**
 * Manages creation and caching of PropertyTarget instances.
 *
 * @internal
 */

const PropertyTargetManager = Object.freeze({
  getOrCreate(source) {
    if (propertyTargetCache.has(source)) {
      /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */
      return propertyTargetCache.get(source);
    }

    let target;

    if (source === defaultElement) {
      target = new RootStyleSheetTarget();
    } else if (source instanceof Document) {
      target = DOM.supportsAdoptedStyleSheets ? new DocumentStyleSheetTarget() : new HeadStyleElementStyleSheetTarget();
    } else if (isFastElement(source)) {
      target = new propertyTargetCtor(source);
    } else {
      target = new ElementStyleSheetTarget(source);
    }

    propertyTargetCache.set(source, target);
    return target;
  }

});

/**
 * Implementation of {@link (DesignToken:interface)}
 */

class DesignTokenImpl extends CSSDirective {
  constructor(configuration) {
    super();
    this.subscribers = new WeakMap();
    this._appliedTo = new Set();
    this.name = configuration.name;

    if (configuration.cssCustomPropertyName !== null) {
      this.cssCustomProperty = `--${configuration.cssCustomPropertyName}`;
      this.cssVar = `var(${this.cssCustomProperty})`;
    }

    this.id = DesignTokenImpl.uniqueId();
    DesignTokenImpl.tokensById.set(this.id, this);
  }

  get appliedTo() {
    return [...this._appliedTo];
  }

  static from(nameOrConfig) {
    return new DesignTokenImpl({
      name: typeof nameOrConfig === "string" ? nameOrConfig : nameOrConfig.name,
      cssCustomPropertyName: typeof nameOrConfig === "string" ? nameOrConfig : nameOrConfig.cssCustomPropertyName === void 0 ? nameOrConfig.name : nameOrConfig.cssCustomPropertyName
    });
  }

  static isCSSDesignToken(token) {
    return typeof token.cssCustomProperty === "string";
  }

  static isDerivedDesignTokenValue(value) {
    return typeof value === "function";
  }
  /**
   * Gets a token by ID. Returns undefined if the token was not found.
   * @param id - The ID of the token
   * @returns
   */


  static getTokenById(id) {
    return DesignTokenImpl.tokensById.get(id);
  }

  getOrCreateSubscriberSet(target = this) {
    return this.subscribers.get(target) || this.subscribers.set(target, new Set()) && this.subscribers.get(target);
  }

  createCSS() {
    return this.cssVar || "";
  }

  getValueFor(element) {
    const value = DesignTokenNode.getOrCreate(element).get(this);

    if (value !== undefined) {
      return value;
    }

    throw new Error(`Value could not be retrieved for token named "${this.name}". Ensure the value is set for ${element} or an ancestor of ${element}.`);
  }

  setValueFor(element, value) {
    this._appliedTo.add(element);

    if (value instanceof DesignTokenImpl) {
      value = this.alias(value);
    }

    DesignTokenNode.getOrCreate(element).set(this, value);
    return this;
  }

  deleteValueFor(element) {
    this._appliedTo.delete(element);

    if (DesignTokenNode.existsFor(element)) {
      DesignTokenNode.getOrCreate(element).delete(this);
    }

    return this;
  }

  withDefault(value) {
    this.setValueFor(defaultElement, value);
    return this;
  }

  subscribe(subscriber, target) {
    const subscriberSet = this.getOrCreateSubscriberSet(target);

    if (target && !DesignTokenNode.existsFor(target)) {
      DesignTokenNode.getOrCreate(target);
    }

    if (!subscriberSet.has(subscriber)) {
      subscriberSet.add(subscriber);
    }
  }

  unsubscribe(subscriber, target) {
    const list = this.subscribers.get(target || this);

    if (list && list.has(subscriber)) {
      list.delete(subscriber);
    }
  }
  /**
   * Notifies subscribers that the value for an element has changed.
   * @param element - The element to emit a notification for
   */


  notify(element) {
    const record = Object.freeze({
      token: this,
      target: element
    });

    if (this.subscribers.has(this)) {
      this.subscribers.get(this).forEach(sub => sub.handleChange(record));
    }

    if (this.subscribers.has(element)) {
      this.subscribers.get(element).forEach(sub => sub.handleChange(record));
    }
  }
  /**
   * Alias the token to the provided token.
   * @param token - the token to alias to
   */


  alias(token) {
    return target => token.getValueFor(target);
  }

}

DesignTokenImpl.uniqueId = (() => {
  let id = 0;
  return () => {
    id++;
    return id.toString(16);
  };
})();
/**
 * Token storage by token ID
 */


DesignTokenImpl.tokensById = new Map();

class CustomPropertyReflector {
  startReflection(token, target) {
    token.subscribe(this, target);
    this.handleChange({
      token,
      target
    });
  }

  stopReflection(token, target) {
    token.unsubscribe(this, target);
    this.remove(token, target);
  }

  handleChange(record) {
    const {
      token,
      target
    } = record;
    this.add(token, target);
  }

  add(token, target) {
    PropertyTargetManager.getOrCreate(target).setProperty(token.cssCustomProperty, this.resolveCSSValue(DesignTokenNode.getOrCreate(target).get(token)));
  }

  remove(token, target) {
    PropertyTargetManager.getOrCreate(target).removeProperty(token.cssCustomProperty);
  }

  resolveCSSValue(value) {
    return value && typeof value.createCSS === "function" ? value.createCSS() : value;
  }

}
/**
 * A light wrapper around BindingObserver to handle value caching and
 * token notification
 */


class DesignTokenBindingObserver {
  constructor(source, token, node) {
    this.source = source;
    this.token = token;
    this.node = node;
    this.dependencies = new Set();
    this.observer = Observable.binding(source, this, false); // This is a little bit hacky because it's using internal APIs of BindingObserverImpl.
    // BindingObserverImpl queues updates to batch it's notifications which doesn't work for this
    // scenario because the DesignToken.getValueFor API is not async. Without this, using DesignToken.getValueFor()
    // after DesignToken.setValueFor() when setting a dependency of the value being retrieved can return a stale
    // value. Assigning .handleChange to .call forces immediate invocation of this classes handleChange() method,
    // allowing resolution of values synchronously.
    // TODO: https://github.com/microsoft/fast/issues/5110

    this.observer.handleChange = this.observer.call;
    this.handleChange();
  }

  disconnect() {
    this.observer.disconnect();
  }
  /**
   * @internal
   */


  handleChange() {
    this.node.store.set(this.token, this.observer.observe(this.node.target, defaultExecutionContext));
  }

}
/**
 * Stores resolved token/value pairs and notifies on changes
 */


class Store {
  constructor() {
    this.values = new Map();
  }

  set(token, value) {
    if (this.values.get(token) !== value) {
      this.values.set(token, value);
      Observable.getNotifier(this).notify(token.id);
    }
  }

  get(token) {
    Observable.track(this, token.id);
    return this.values.get(token);
  }

  delete(token) {
    this.values.delete(token);
  }

  all() {
    return this.values.entries();
  }

}

const nodeCache = new WeakMap();
const childToParent = new WeakMap();
/**
 * A node responsible for setting and getting token values,
 * emitting values to CSS custom properties, and maintaining
 * inheritance structures.
 */

class DesignTokenNode {
  constructor(target) {
    this.target = target;
    /**
     * Stores all resolved token values for a node
     */

    this.store = new Store();
    /**
     * All children assigned to the node
     */

    this.children = [];
    /**
     * All values explicitly assigned to the node in their raw form
     */

    this.assignedValues = new Map();
    /**
     * Tokens currently being reflected to CSS custom properties
     */

    this.reflecting = new Set();
    /**
     * Binding observers for assigned and inherited derived values.
     */

    this.bindingObservers = new Map();
    /**
     * Emits notifications to token when token values
     * change the DesignTokenNode
     */

    this.tokenValueChangeHandler = {
      handleChange: (source, arg) => {
        const token = DesignTokenImpl.getTokenById(arg);

        if (token) {
          // Notify any token subscribers
          token.notify(this.target);

          if (DesignTokenImpl.isCSSDesignToken(token)) {
            const parent = this.parent;
            const reflecting = this.isReflecting(token);

            if (parent) {
              const parentValue = parent.get(token);
              const sourceValue = source.get(token);

              if (parentValue !== sourceValue && !reflecting) {
                this.reflectToCSS(token);
              } else if (parentValue === sourceValue && reflecting) {
                this.stopReflectToCSS(token);
              }
            } else if (!reflecting) {
              this.reflectToCSS(token);
            }
          }
        }
      }
    };
    nodeCache.set(target, this); // Map store change notifications to token change notifications

    Observable.getNotifier(this.store).subscribe(this.tokenValueChangeHandler);

    if (target instanceof FASTElement) {
      target.$fastController.addBehaviors([this]);
    } else if (target.isConnected) {
      this.bind();
    }
  }
  /**
   * Returns a DesignTokenNode for an element.
   * Creates a new instance if one does not already exist for a node,
   * otherwise returns the cached instance
   *
   * @param target - The HTML element to retrieve a DesignTokenNode for
   */


  static getOrCreate(target) {
    return nodeCache.get(target) || new DesignTokenNode(target);
  }
  /**
   * Determines if a DesignTokenNode has been created for a target
   * @param target - The element to test
   */


  static existsFor(target) {
    return nodeCache.has(target);
  }
  /**
   * Searches for and return the nearest parent DesignTokenNode.
   * Null is returned if no node is found or the node provided is for a default element.
   */


  static findParent(node) {
    if (!(defaultElement === node.target)) {
      let parent = composedParent(node.target);

      while (parent !== null) {
        if (nodeCache.has(parent)) {
          return nodeCache.get(parent);
        }

        parent = composedParent(parent);
      }

      return DesignTokenNode.getOrCreate(defaultElement);
    }

    return null;
  }
  /**
   * Finds the closest node with a value explicitly assigned for a token, otherwise null.
   * @param token - The token to look for
   * @param start - The node to start looking for value assignment
   * @returns
   */


  static findClosestAssignedNode(token, start) {
    let current = start;

    do {
      if (current.has(token)) {
        return current;
      }

      current = current.parent ? current.parent : current.target !== defaultElement ? DesignTokenNode.getOrCreate(defaultElement) : null;
    } while (current !== null);

    return null;
  }
  /**
   * The parent DesignTokenNode, or null.
   */


  get parent() {
    return childToParent.get(this) || null;
  }
  /**
   * Checks if a token has been assigned an explicit value the node.
   * @param token - the token to check.
   */


  has(token) {
    return this.assignedValues.has(token);
  }
  /**
   * Gets the value of a token for a node
   * @param token - The token to retrieve the value for
   * @returns
   */


  get(token) {
    const value = this.store.get(token);

    if (value !== undefined) {
      return value;
    }

    const raw = this.getRaw(token);

    if (raw !== undefined) {
      this.hydrate(token, raw);
      return this.get(token);
    }
  }
  /**
   * Retrieves the raw assigned value of a token from the nearest assigned node.
   * @param token - The token to retrieve a raw value for
   * @returns
   */


  getRaw(token) {
    var _a;

    if (this.assignedValues.has(token)) {
      return this.assignedValues.get(token);
    }

    return (_a = DesignTokenNode.findClosestAssignedNode(token, this)) === null || _a === void 0 ? void 0 : _a.getRaw(token);
  }
  /**
   * Sets a token to a value for a node
   * @param token - The token to set
   * @param value - The value to set the token to
   */


  set(token, value) {
    if (DesignTokenImpl.isDerivedDesignTokenValue(this.assignedValues.get(token))) {
      this.tearDownBindingObserver(token);
    }

    this.assignedValues.set(token, value);

    if (DesignTokenImpl.isDerivedDesignTokenValue(value)) {
      this.setupBindingObserver(token, value);
    } else {
      this.store.set(token, value);
    }
  }
  /**
   * Deletes a token value for the node.
   * @param token - The token to delete the value for
   */


  delete(token) {
    this.assignedValues.delete(token);
    this.tearDownBindingObserver(token);
    const upstream = this.getRaw(token);

    if (upstream) {
      this.hydrate(token, upstream);
    } else {
      this.store.delete(token);
    }
  }
  /**
   * Invoked when the DesignTokenNode.target is attached to the document
   */


  bind() {
    const parent = DesignTokenNode.findParent(this);

    if (parent) {
      parent.appendChild(this);
    }

    for (const key of this.assignedValues.keys()) {
      key.notify(this.target);
    }
  }
  /**
   * Invoked when the DesignTokenNode.target is detached from the document
   */


  unbind() {
    if (this.parent) {
      const parent = childToParent.get(this);
      parent.removeChild(this);
    }
  }
  /**
   * Appends a child to a parent DesignTokenNode.
   * @param child - The child to append to the node
   */


  appendChild(child) {
    if (child.parent) {
      childToParent.get(child).removeChild(child);
    }

    const reParent = this.children.filter(x => child.contains(x));
    childToParent.set(child, this);
    this.children.push(child);
    reParent.forEach(x => child.appendChild(x));
    Observable.getNotifier(this.store).subscribe(child); // How can we not notify *every* subscriber?

    for (const [token, value] of this.store.all()) {
      child.hydrate(token, this.bindingObservers.has(token) ? this.getRaw(token) : value);
    }
  }
  /**
   * Removes a child from a node.
   * @param child - The child to remove.
   */


  removeChild(child) {
    const childIndex = this.children.indexOf(child);

    if (childIndex !== -1) {
      this.children.splice(childIndex, 1);
    }

    Observable.getNotifier(this.store).unsubscribe(child);
    return child.parent === this ? childToParent.delete(child) : false;
  }
  /**
   * Tests whether a provided node is contained by
   * the calling node.
   * @param test - The node to test
   */


  contains(test) {
    return composedContains(this.target, test.target);
  }
  /**
   * Instructs the node to reflect a design token for the provided token.
   * @param token - The design token to reflect
   */


  reflectToCSS(token) {
    if (!this.isReflecting(token)) {
      this.reflecting.add(token);
      DesignTokenNode.cssCustomPropertyReflector.startReflection(token, this.target);
    }
  }
  /**
   * Stops reflecting a DesignToken to CSS
   * @param token - The design token to stop reflecting
   */


  stopReflectToCSS(token) {
    if (this.isReflecting(token)) {
      this.reflecting.delete(token);
      DesignTokenNode.cssCustomPropertyReflector.stopReflection(token, this.target);
    }
  }
  /**
   * Determines if a token is being reflected to CSS for a node.
   * @param token - The token to check for reflection
   * @returns
   */


  isReflecting(token) {
    return this.reflecting.has(token);
  }
  /**
   * Handle changes to upstream tokens
   * @param source - The parent DesignTokenNode
   * @param property - The token ID that changed
   */


  handleChange(source, property) {
    const token = DesignTokenImpl.getTokenById(property);

    if (!token) {
      return;
    }

    this.hydrate(token, this.getRaw(token));
  }
  /**
   * Hydrates a token with a DesignTokenValue, making retrieval available.
   * @param token - The token to hydrate
   * @param value - The value to hydrate
   */


  hydrate(token, value) {
    if (!this.has(token)) {
      const observer = this.bindingObservers.get(token);

      if (DesignTokenImpl.isDerivedDesignTokenValue(value)) {
        if (observer) {
          // If the binding source doesn't match, we need
          // to update the binding
          if (observer.source !== value) {
            this.tearDownBindingObserver(token);
            this.setupBindingObserver(token, value);
          }
        } else {
          this.setupBindingObserver(token, value);
        }
      } else {
        if (observer) {
          this.tearDownBindingObserver(token);
        }

        this.store.set(token, value);
      }
    }
  }
  /**
   * Sets up a binding observer for a derived token value that notifies token
   * subscribers on change.
   *
   * @param token - The token to notify when the binding updates
   * @param source - The binding source
   */


  setupBindingObserver(token, source) {
    const binding = new DesignTokenBindingObserver(source, token, this);
    this.bindingObservers.set(token, binding);
    return binding;
  }
  /**
   * Tear down a binding observer for a token.
   */


  tearDownBindingObserver(token) {
    if (this.bindingObservers.has(token)) {
      this.bindingObservers.get(token).disconnect();
      this.bindingObservers.delete(token);
      return true;
    }

    return false;
  }

}
/**
 * Responsible for reflecting tokens to CSS custom properties
 */


DesignTokenNode.cssCustomPropertyReflector = new CustomPropertyReflector();

__decorate$1([observable], DesignTokenNode.prototype, "children", void 0);

function create$2(nameOrConfig) {
  return DesignTokenImpl.from(nameOrConfig);
}
/* eslint-enable @typescript-eslint/no-unused-vars */

/**
 * Factory object for creating {@link (DesignToken:interface)} instances.
 * @public
 */


const DesignToken = Object.freeze({
  create: create$2,

  /**
   * Informs DesignToken that an HTMLElement for which tokens have
   * been set has been connected to the document.
   *
   * The browser does not provide a reliable mechanism to observe an HTMLElement's connectedness
   * in all scenarios, so invoking this method manually is necessary when:
   *
   * 1. Token values are set for an HTMLElement.
   * 2. The HTMLElement does not inherit from FASTElement.
   * 3. The HTMLElement is not connected to the document when token values are set.
   *
   * @param element - The element to notify
   * @returns - true if notification was successful, otherwise false.
   */
  notifyConnection(element) {
    if (!element.isConnected || !DesignTokenNode.existsFor(element)) {
      return false;
    }

    DesignTokenNode.getOrCreate(element).bind();
    return true;
  },

  /**
   * Informs DesignToken that an HTMLElement for which tokens have
   * been set has been disconnected to the document.
   *
   * The browser does not provide a reliable mechanism to observe an HTMLElement's connectedness
   * in all scenarios, so invoking this method manually is necessary when:
   *
   * 1. Token values are set for an HTMLElement.
   * 2. The HTMLElement does not inherit from FASTElement.
   *
   * @param element - The element to notify
   * @returns - true if notification was successful, otherwise false.
   */
  notifyDisconnection(element) {
    if (element.isConnected || !DesignTokenNode.existsFor(element)) {
      return false;
    }

    DesignTokenNode.getOrCreate(element).unbind();
    return true;
  },

  /**
   * Registers and element or document as a DesignToken root.
   * {@link CSSDesignToken | CSSDesignTokens} with default values assigned via
   * {@link (DesignToken:interface).withDefault} will emit CSS custom properties to all
   * registered roots.
   * @param target - The root to register
   */
  registerRoot(target = defaultElement) {
    RootStyleSheetTarget.registerRoot(target);
  },

  /**
   * Unregister an element or document as a DesignToken root.
   * @param target - The root to deregister
   */
  unregisterRoot(target = defaultElement) {
    RootStyleSheetTarget.unregisterRoot(target);
  }

});
/* eslint-enable @typescript-eslint/no-non-null-assertion */

/* eslint-disable @typescript-eslint/no-non-null-assertion */

/**
 * Indicates what to do with an ambiguous (duplicate) element.
 * @public
 */

const ElementDisambiguation = Object.freeze({
  /**
   * Skip defining the element but still call the provided callback passed
   * to DesignSystemRegistrationContext.tryDefineElement
   */
  definitionCallbackOnly: null,

  /**
   * Ignore the duplicate element entirely.
   */
  ignoreDuplicate: Symbol()
});
const elementTypesByTag = new Map();
const elementTagsByType = new Map();
let rootDesignSystem = null;
const designSystemKey = DI.createInterface(x => x.cachedCallback(handler => {
  if (rootDesignSystem === null) {
    rootDesignSystem = new DefaultDesignSystem(null, handler);
  }

  return rootDesignSystem;
}));
/**
 * An API gateway to design system features.
 * @public
 */

const DesignSystem = Object.freeze({
  /**
   * Returns the HTML element name that the type is defined as.
   * @param type - The type to lookup.
   * @public
   */
  tagFor(type) {
    return elementTagsByType.get(type);
  },

  /**
   * Searches the DOM hierarchy for the design system that is responsible
   * for the provided element.
   * @param element - The element to locate the design system for.
   * @returns The located design system.
   * @public
   */
  responsibleFor(element) {
    const owned = element.$$designSystem$$;

    if (owned) {
      return owned;
    }

    const container = DI.findResponsibleContainer(element);
    return container.get(designSystemKey);
  },

  /**
   * Gets the DesignSystem if one is explicitly defined on the provided element;
   * otherwise creates a design system defined directly on the element.
   * @param element - The element to get or create a design system for.
   * @returns The design system.
   * @public
   */
  getOrCreate(node) {
    if (!node) {
      if (rootDesignSystem === null) {
        rootDesignSystem = DI.getOrCreateDOMContainer().get(designSystemKey);
      }

      return rootDesignSystem;
    }

    const owned = node.$$designSystem$$;

    if (owned) {
      return owned;
    }

    const container = DI.getOrCreateDOMContainer(node);

    if (container.has(designSystemKey, false)) {
      return container.get(designSystemKey);
    } else {
      const system = new DefaultDesignSystem(node, container);
      container.register(Registration.instance(designSystemKey, system));
      return system;
    }
  }

});

function extractTryDefineElementParams(params, elementDefinitionType, elementDefinitionCallback) {
  if (typeof params === "string") {
    return {
      name: params,
      type: elementDefinitionType,
      callback: elementDefinitionCallback
    };
  } else {
    return params;
  }
}

class DefaultDesignSystem {
  constructor(owner, container) {
    this.owner = owner;
    this.container = container;
    this.designTokensInitialized = false;
    this.prefix = "fast";
    this.shadowRootMode = undefined;

    this.disambiguate = () => ElementDisambiguation.definitionCallbackOnly;

    if (owner !== null) {
      owner.$$designSystem$$ = this;
    }
  }

  withPrefix(prefix) {
    this.prefix = prefix;
    return this;
  }

  withShadowRootMode(mode) {
    this.shadowRootMode = mode;
    return this;
  }

  withElementDisambiguation(callback) {
    this.disambiguate = callback;
    return this;
  }

  withDesignTokenRoot(root) {
    this.designTokenRoot = root;
    return this;
  }

  register(...registrations) {
    const container = this.container;
    const elementDefinitionEntries = [];
    const disambiguate = this.disambiguate;
    const shadowRootMode = this.shadowRootMode;
    const context = {
      elementPrefix: this.prefix,

      tryDefineElement(params, elementDefinitionType, elementDefinitionCallback) {
        const extractedParams = extractTryDefineElementParams(params, elementDefinitionType, elementDefinitionCallback);
        const {
          name,
          callback,
          baseClass
        } = extractedParams;
        let {
          type
        } = extractedParams;
        let elementName = name;
        let typeFoundByName = elementTypesByTag.get(elementName);
        let needsDefine = true;

        while (typeFoundByName) {
          const result = disambiguate(elementName, type, typeFoundByName);

          switch (result) {
            case ElementDisambiguation.ignoreDuplicate:
              return;

            case ElementDisambiguation.definitionCallbackOnly:
              needsDefine = false;
              typeFoundByName = void 0;
              break;

            default:
              elementName = result;
              typeFoundByName = elementTypesByTag.get(elementName);
              break;
          }
        }

        if (needsDefine) {
          if (elementTagsByType.has(type) || type === FoundationElement) {
            type = class extends type {};
          }

          elementTypesByTag.set(elementName, type);
          elementTagsByType.set(type, elementName);

          if (baseClass) {
            elementTagsByType.set(baseClass, elementName);
          }
        }

        elementDefinitionEntries.push(new ElementDefinitionEntry(container, elementName, type, shadowRootMode, callback, needsDefine));
      }

    };

    if (!this.designTokensInitialized) {
      this.designTokensInitialized = true;

      if (this.designTokenRoot !== null) {
        DesignToken.registerRoot(this.designTokenRoot);
      }
    }

    container.registerWithContext(context, ...registrations);

    for (const entry of elementDefinitionEntries) {
      entry.callback(entry);

      if (entry.willDefine && entry.definition !== null) {
        entry.definition.define();
      }
    }

    return this;
  }

}

class ElementDefinitionEntry {
  constructor(container, name, type, shadowRootMode, callback, willDefine) {
    this.container = container;
    this.name = name;
    this.type = type;
    this.shadowRootMode = shadowRootMode;
    this.callback = callback;
    this.willDefine = willDefine;
    this.definition = null;
  }

  definePresentation(presentation) {
    ComponentPresentation.define(this.name, presentation, this.container);
  }

  defineElement(definition) {
    this.definition = new FASTElementDefinition(this.type, Object.assign(Object.assign({}, definition), {
      name: this.name
    }));
  }

  tagFor(type) {
    return DesignSystem.tagFor(type);
  }

}
/* eslint-enable @typescript-eslint/no-non-null-assertion */

/**
 * The template for the {@link @microsoft/fast-foundation#Dialog} component.
 * @public
 */

const dialogTemplate = (context, definition) => html`<div class="positioning-region" part="positioning-region">${when(x => x.modal, html`<div class="overlay" part="overlay" role="presentation" @click="${x => x.dismiss()}"></div>`)}<div role="dialog" tabindex="-1" class="control" part="control" aria-modal="${x => x.modal}" aria-describedby="${x => x.ariaDescribedby}" aria-labelledby="${x => x.ariaLabelledby}" aria-label="${x => x.ariaLabel}" ${ref("dialog")}><slot></slot></div></div>`;

/*!
* tabbable 5.2.1
* @license MIT, https://github.com/focus-trap/tabbable/blob/master/LICENSE
*/
var candidateSelectors = ['input', 'select', 'textarea', 'a[href]', 'button', '[tabindex]', 'audio[controls]', 'video[controls]', '[contenteditable]:not([contenteditable="false"])', 'details>summary:first-of-type', 'details'];
var candidateSelector = /* #__PURE__ */candidateSelectors.join(',');
var matches = typeof Element === 'undefined' ? function () {} : Element.prototype.matches || Element.prototype.msMatchesSelector || Element.prototype.webkitMatchesSelector;

var isContentEditable = function isContentEditable(node) {
  return node.contentEditable === 'true';
};

var getTabindex = function getTabindex(node) {
  var tabindexAttr = parseInt(node.getAttribute('tabindex'), 10);

  if (!isNaN(tabindexAttr)) {
    return tabindexAttr;
  } // Browsers do not return `tabIndex` correctly for contentEditable nodes;
  // so if they don't have a tabindex attribute specifically set, assume it's 0.


  if (isContentEditable(node)) {
    return 0;
  } // in Chrome, <details/>, <audio controls/> and <video controls/> elements get a default
  //  `tabIndex` of -1 when the 'tabindex' attribute isn't specified in the DOM,
  //  yet they are still part of the regular tab order; in FF, they get a default
  //  `tabIndex` of 0; since Chrome still puts those elements in the regular tab
  //  order, consider their tab index to be 0.


  if ((node.nodeName === 'AUDIO' || node.nodeName === 'VIDEO' || node.nodeName === 'DETAILS') && node.getAttribute('tabindex') === null) {
    return 0;
  }

  return node.tabIndex;
};

var isInput = function isInput(node) {
  return node.tagName === 'INPUT';
};

var isHiddenInput = function isHiddenInput(node) {
  return isInput(node) && node.type === 'hidden';
};

var isDetailsWithSummary = function isDetailsWithSummary(node) {
  var r = node.tagName === 'DETAILS' && Array.prototype.slice.apply(node.children).some(function (child) {
    return child.tagName === 'SUMMARY';
  });
  return r;
};

var getCheckedRadio = function getCheckedRadio(nodes, form) {
  for (var i = 0; i < nodes.length; i++) {
    if (nodes[i].checked && nodes[i].form === form) {
      return nodes[i];
    }
  }
};

var isTabbableRadio = function isTabbableRadio(node) {
  if (!node.name) {
    return true;
  }

  var radioScope = node.form || node.ownerDocument;

  var queryRadios = function queryRadios(name) {
    return radioScope.querySelectorAll('input[type="radio"][name="' + name + '"]');
  };

  var radioSet;

  if (typeof window !== 'undefined' && typeof window.CSS !== 'undefined' && typeof window.CSS.escape === 'function') {
    radioSet = queryRadios(window.CSS.escape(node.name));
  } else {
    try {
      radioSet = queryRadios(node.name);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('Looks like you have a radio button with a name attribute containing invalid CSS selector characters and need the CSS.escape polyfill: %s', err.message);
      return false;
    }
  }

  var checked = getCheckedRadio(radioSet, node.form);
  return !checked || checked === node;
};

var isRadio = function isRadio(node) {
  return isInput(node) && node.type === 'radio';
};

var isNonTabbableRadio = function isNonTabbableRadio(node) {
  return isRadio(node) && !isTabbableRadio(node);
};

var isHidden = function isHidden(node, displayCheck) {
  if (getComputedStyle(node).visibility === 'hidden') {
    return true;
  }

  var isDirectSummary = matches.call(node, 'details>summary:first-of-type');
  var nodeUnderDetails = isDirectSummary ? node.parentElement : node;

  if (matches.call(nodeUnderDetails, 'details:not([open]) *')) {
    return true;
  }

  if (!displayCheck || displayCheck === 'full') {
    while (node) {
      if (getComputedStyle(node).display === 'none') {
        return true;
      }

      node = node.parentElement;
    }
  } else if (displayCheck === 'non-zero-area') {
    var _node$getBoundingClie = node.getBoundingClientRect(),
        width = _node$getBoundingClie.width,
        height = _node$getBoundingClie.height;

    return width === 0 && height === 0;
  }

  return false;
}; // form fields (nested) inside a disabled fieldset are not focusable/tabbable
//  unless they are in the _first_ <legend> element of the top-most disabled
//  fieldset


var isDisabledFromFieldset = function isDisabledFromFieldset(node) {
  if (isInput(node) || node.tagName === 'SELECT' || node.tagName === 'TEXTAREA' || node.tagName === 'BUTTON') {
    var parentNode = node.parentElement;

    while (parentNode) {
      if (parentNode.tagName === 'FIELDSET' && parentNode.disabled) {
        // look for the first <legend> as an immediate child of the disabled
        //  <fieldset>: if the node is in that legend, it'll be enabled even
        //  though the fieldset is disabled; otherwise, the node is in a
        //  secondary/subsequent legend, or somewhere else within the fieldset
        //  (however deep nested) and it'll be disabled
        for (var i = 0; i < parentNode.children.length; i++) {
          var child = parentNode.children.item(i);

          if (child.tagName === 'LEGEND') {
            if (child.contains(node)) {
              return false;
            } // the node isn't in the first legend (in doc order), so no matter
            //  where it is now, it'll be disabled


            return true;
          }
        } // the node isn't in a legend, so no matter where it is now, it'll be disabled


        return true;
      }

      parentNode = parentNode.parentElement;
    }
  } // else, node's tabbable/focusable state should not be affected by a fieldset's
  //  enabled/disabled state


  return false;
};

var isNodeMatchingSelectorFocusable = function isNodeMatchingSelectorFocusable(options, node) {
  if (node.disabled || isHiddenInput(node) || isHidden(node, options.displayCheck) || // For a details element with a summary, the summary element gets the focus
  isDetailsWithSummary(node) || isDisabledFromFieldset(node)) {
    return false;
  }

  return true;
};

var isNodeMatchingSelectorTabbable = function isNodeMatchingSelectorTabbable(options, node) {
  if (!isNodeMatchingSelectorFocusable(options, node) || isNonTabbableRadio(node) || getTabindex(node) < 0) {
    return false;
  }

  return true;
};

var isTabbable = function isTabbable(node, options) {
  options = options || {};

  if (!node) {
    throw new Error('No node provided');
  }

  if (matches.call(node, candidateSelector) === false) {
    return false;
  }

  return isNodeMatchingSelectorTabbable(options, node);
};

var focusableCandidateSelector = /* #__PURE__ */candidateSelectors.concat('iframe').join(',');

var isFocusable = function isFocusable(node, options) {
  options = options || {};

  if (!node) {
    throw new Error('No node provided');
  }

  if (matches.call(node, focusableCandidateSelector) === false) {
    return false;
  }

  return isNodeMatchingSelectorFocusable(options, node);
};

/**
 * A Switch Custom HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#dialog | ARIA dialog }.
 *
 * @slot - The default slot for the dialog content
 * @csspart positioning-region - A wrapping element used to center the dialog and position the modal overlay
 * @csspart overlay - The modal dialog overlay
 * @csspart control - The dialog element
 * @fires cancel - Fires a custom 'cancel' event when the modal overlay is clicked
 * @fires close - Fires a custom 'close' event when the dialog is hidden
 *
 * @public
 */

class Dialog extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * Indicates the element is modal. When modal, user mouse interaction will be limited to the contents of the element by a modal
     * overlay.  Clicks on the overlay will cause the dialog to emit a "dismiss" event.
     * @public
     * @defaultValue - true
     * @remarks
     * HTML Attribute: modal
     */

    this.modal = true;
    /**
     * The hidden state of the element.
     *
     * @public
     * @defaultValue - false
     * @remarks
     * HTML Attribute: hidden
     */

    this.hidden = false;
    /**
     * Indicates that the dialog should trap focus.
     *
     * @public
     * @defaultValue - true
     * @remarks
     * HTML Attribute: trap-focus
     */

    this.trapFocus = true;

    this.trapFocusChanged = () => {
      if (this.$fastController.isConnected) {
        this.updateTrapFocus();
      }
    };
    /**
     * @internal
     */


    this.isTrappingFocus = false;

    this.handleDocumentKeydown = e => {
      if (!e.defaultPrevented && !this.hidden) {
        switch (e.key) {
          case keyEscape:
            this.dismiss();
            e.preventDefault();
            break;

          case keyTab:
            this.handleTabKeyDown(e);
            break;
        }
      }
    };

    this.handleDocumentFocus = e => {
      if (!e.defaultPrevented && this.shouldForceFocus(e.target)) {
        this.focusFirstElement();
        e.preventDefault();
      }
    };

    this.handleTabKeyDown = e => {
      if (!this.trapFocus || this.hidden) {
        return;
      }

      const bounds = this.getTabQueueBounds();

      if (bounds.length === 0) {
        return;
      }

      if (bounds.length === 1) {
        // keep focus on single element
        bounds[0].focus();
        e.preventDefault();
        return;
      }

      if (e.shiftKey && e.target === bounds[0]) {
        bounds[bounds.length - 1].focus();
        e.preventDefault();
      } else if (!e.shiftKey && e.target === bounds[bounds.length - 1]) {
        bounds[0].focus();
        e.preventDefault();
      }

      return;
    };

    this.getTabQueueBounds = () => {
      const bounds = [];
      return Dialog.reduceTabbableItems(bounds, this);
    };
    /**
     * focus on first element of tab queue
     */


    this.focusFirstElement = () => {
      const bounds = this.getTabQueueBounds();

      if (bounds.length > 0) {
        bounds[0].focus();
      } else {
        if (this.dialog instanceof HTMLElement) {
          this.dialog.focus();
        }
      }
    };
    /**
     * we should only focus if focus has not already been brought to the dialog
     */


    this.shouldForceFocus = currentFocusElement => {
      return this.isTrappingFocus && !this.contains(currentFocusElement);
    };
    /**
     * we should we be active trapping focus
     */


    this.shouldTrapFocus = () => {
      return this.trapFocus && !this.hidden;
    };
    /**
     *
     *
     * @internal
     */


    this.updateTrapFocus = shouldTrapFocusOverride => {
      const shouldTrapFocus = shouldTrapFocusOverride === undefined ? this.shouldTrapFocus() : shouldTrapFocusOverride;

      if (shouldTrapFocus && !this.isTrappingFocus) {
        this.isTrappingFocus = true; // Add an event listener for focusin events if we are trapping focus

        document.addEventListener("focusin", this.handleDocumentFocus);
        DOM.queueUpdate(() => {
          if (this.shouldForceFocus(document.activeElement)) {
            this.focusFirstElement();
          }
        });
      } else if (!shouldTrapFocus && this.isTrappingFocus) {
        this.isTrappingFocus = false; // remove event listener if we are not trapping focus

        document.removeEventListener("focusin", this.handleDocumentFocus);
      }
    };
  }
  /**
   * @internal
   */


  dismiss() {
    this.$emit("dismiss"); // implement `<dialog>` interface

    this.$emit("cancel");
  }
  /**
   * The method to show the dialog.
   *
   * @public
   */


  show() {
    this.hidden = false;
  }
  /**
   * The method to hide the dialog.
   *
   * @public
   */


  hide() {
    this.hidden = true; // implement `<dialog>` interface

    this.$emit("close");
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    document.addEventListener("keydown", this.handleDocumentKeydown);
    this.notifier = Observable.getNotifier(this);
    this.notifier.subscribe(this, "hidden");
    this.updateTrapFocus();
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    super.disconnectedCallback(); // remove keydown event listener

    document.removeEventListener("keydown", this.handleDocumentKeydown); // if we are trapping focus remove the focusin listener

    this.updateTrapFocus(false);
    this.notifier.unsubscribe(this, "hidden");
  }
  /**
   * @internal
   */


  handleChange(source, propertyName) {
    switch (propertyName) {
      case "hidden":
        this.updateTrapFocus();
        break;
    }
  }
  /**
   * Reduce a collection to only its focusable elements.
   *
   * @param elements - Collection of elements to reduce
   * @param element - The current element
   *
   * @internal
   */


  static reduceTabbableItems(elements, element) {
    if (element.getAttribute("tabindex") === "-1") {
      return elements;
    }

    if (isTabbable(element) || Dialog.isFocusableFastElement(element) && Dialog.hasTabbableShadow(element)) {
      elements.push(element);
      return elements;
    }

    if (element.childElementCount) {
      return elements.concat(Array.from(element.children).reduce(Dialog.reduceTabbableItems, []));
    }

    return elements;
  }
  /**
   * Test if element is focusable fast element
   *
   * @param element - The element to check
   *
   * @internal
   */


  static isFocusableFastElement(element) {
    var _a, _b;

    return !!((_b = (_a = element.$fastController) === null || _a === void 0 ? void 0 : _a.definition.shadowOptions) === null || _b === void 0 ? void 0 : _b.delegatesFocus);
  }
  /**
   * Test if the element has a focusable shadow
   *
   * @param element - The element to check
   *
   * @internal
   */


  static hasTabbableShadow(element) {
    var _a, _b;

    return Array.from((_b = (_a = element.shadowRoot) === null || _a === void 0 ? void 0 : _a.querySelectorAll("*")) !== null && _b !== void 0 ? _b : []).some(x => {
      return isTabbable(x);
    });
  }

}

__decorate$1([attr({
  mode: "boolean"
})], Dialog.prototype, "modal", void 0);

__decorate$1([attr({
  mode: "boolean"
})], Dialog.prototype, "hidden", void 0);

__decorate$1([attr({
  attribute: "trap-focus",
  mode: "boolean"
})], Dialog.prototype, "trapFocus", void 0);

__decorate$1([attr({
  attribute: "aria-describedby"
})], Dialog.prototype, "ariaDescribedby", void 0);

__decorate$1([attr({
  attribute: "aria-labelledby"
})], Dialog.prototype, "ariaLabelledby", void 0);

__decorate$1([attr({
  attribute: "aria-label"
})], Dialog.prototype, "ariaLabel", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#Disclosure} component.
 * @public
 */

const disclosureTemplate = (context, definition) => html`<details class="disclosure" ${ref("details")}><summary class="invoker" role="button" aria-controls="disclosure-content" aria-expanded="${x => x.expanded}"><slot name="start"></slot><slot name="title">${x => x.title}</slot><slot name="end"></slot></summary><div id="disclosure-content"><slot></slot></div></details>`;

/**
 * A Disclosure Custom HTML Element.
 * Based largely on the {@link https://w3c.github.io/aria-practices/#disclosure | disclosure element }.
 *
 * @slot start - Content which can be provided before the summary content
 * @slot end - Content which can be provided after the summary content
 * @slot title - The summary content
 * @slot - The default slot for the disclosure content
 * @fires toggle - fires a toggle event when the summary is toggled
 *
 * @public
 */

class Disclosure$1 extends FoundationElement {
  /**
   * @internal
   */
  connectedCallback() {
    super.connectedCallback();
    this.setup();
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    super.disconnectedCallback();
    this.details.removeEventListener("toggle", this.onToggle);
  }
  /**
   * Show extra content.
   */


  show() {
    this.details.open = true;
  }
  /**
   * Hide extra content.
   */


  hide() {
    this.details.open = false;
  }
  /**
   * Toggle the current(expanded/collapsed) state.
   */


  toggle() {
    this.details.open = !this.details.open;
  }
  /**
   * Register listener and set default disclosure mode
   */


  setup() {
    this.onToggle = this.onToggle.bind(this);
    this.details.addEventListener("toggle", this.onToggle);

    if (this.expanded) {
      this.show();
    }
  }
  /**
   * Update the aria attr and fire `toggle` event
   */


  onToggle() {
    this.expanded = this.details.open;
    this.$emit("toggle");
  }

}

__decorate$1([attr({
  mode: "boolean"
})], Disclosure$1.prototype, "expanded", void 0);

__decorate$1([attr], Disclosure$1.prototype, "title", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#Divider} component.
 * @public
 */

const dividerTemplate = (context, definition) => html`<template role="${x => x.role}" aria-orientation="${x => x.orientation}"></template>`;

/**
 * Divider roles
 * @public
 */
const DividerRole = {
  /**
   * The divider semantically separates content
   */
  separator: "separator",

  /**
   * The divider has no semantic value and is for visual presentation only.
   */
  presentation: "presentation"
};

/**
 * A Divider Custom HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#separator | ARIA separator } or {@link https://www.w3.org/TR/wai-aria-1.1/#presentation | ARIA presentation}.
 *
 * @public
 */

class Divider extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * The role of the element.
     *
     * @public
     * @remarks
     * HTML Attribute: role
     */

    this.role = DividerRole.separator;
    /**
     * The orientation of the divider.
     *
     * @public
     * @remarks
     * HTML Attribute: orientation
     */

    this.orientation = Orientation.horizontal;
  }

}

__decorate$1([attr], Divider.prototype, "role", void 0);

__decorate$1([attr], Divider.prototype, "orientation", void 0);

/**
 * The direction options for flipper.
 * @public
 */
const FlipperDirection = {
  next: "next",
  previous: "previous"
};

/**
 * The template for the {@link @microsoft/fast-foundation#Flipper} component.
 * @public
 */

const flipperTemplate = (context, definition) => html`<template role="button" aria-disabled="${x => x.disabled ? true : void 0}" tabindex="${x => x.hiddenFromAT ? -1 : 0}" class="${x => x.direction} ${x => x.disabled ? "disabled" : ""}" @keyup="${(x, c) => x.keyupHandler(c.event)}">${when(x => x.direction === FlipperDirection.next, html`<span part="next" class="next"><slot name="next">${definition.next || ""}</slot></span>`)} ${when(x => x.direction === FlipperDirection.previous, html`<span part="previous" class="previous"><slot name="previous">${definition.previous || ""}</slot></span>`)}</template>`;

/**
 * A Flipper Custom HTML Element.
 * Flippers are a form of button that implies directional content navigation, such as in a carousel.
 *
 * @slot next - The next flipper content
 * @slot previous - The previous flipper content
 * @csspart next - Wraps the next flipper content
 * @csspart previous - Wraps the previous flipper content
 * @fires click - Fires a custom 'click' event when Enter or Space is invoked via keyboard and the flipper is exposed to assistive technologies.
 *
 * @public
 */

class Flipper extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * Indicates the flipper should be hidden from assistive technology. Because flippers are often supplementary navigation, they are often hidden from assistive technology.
     *
     * @public
     * @defaultValue - true
     * @remarks
     * HTML Attribute: aria-hidden
     */

    this.hiddenFromAT = true;
    /**
     * The direction that the flipper implies navigating.
     *
     * @public
     * @remarks
     * HTML Attribute: direction
     */

    this.direction = FlipperDirection.next;
  }
  /**
   * Simulate a click event when the flipper has focus and the user hits enter or space keys
   * Blur focus if the user hits escape key
   * @param e - Keyboard event
   * @public
   */


  keyupHandler(e) {
    if (!this.hiddenFromAT) {
      const key = e.key;

      if (key === "Enter" || key === "Space") {
        this.$emit("click", e);
      }

      if (key === "Escape") {
        this.blur();
      }
    }
  }

}

__decorate$1([attr({
  mode: "boolean"
})], Flipper.prototype, "disabled", void 0);

__decorate$1([attr({
  attribute: "aria-hidden",
  converter: booleanConverter
})], Flipper.prototype, "hiddenFromAT", void 0);

__decorate$1([attr], Flipper.prototype, "direction", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(ListboxOption:class)} component.
 * @public
 */

const listboxOptionTemplate = (context, definition) => html`<template aria-checked="${x => x.ariaChecked}" aria-disabled="${x => x.ariaDisabled}" aria-posinset="${x => x.ariaPosInSet}" aria-selected="${x => x.ariaSelected}" aria-setsize="${x => x.ariaSetSize}" class="${x => [x.checked && "checked", x.selected && "selected", x.disabled && "disabled"].filter(Boolean).join(" ")}" role="option">${startSlotTemplate(context, definition)}<span class="content" part="content"><slot ${slotted("content")}></slot></span>${endSlotTemplate(context, definition)}</template>`;

/**
 * A Listbox Custom HTML Element.
 * Implements the {@link https://w3c.github.io/aria/#listbox | ARIA listbox }.
 *
 * @public
 */

class ListboxElement extends Listbox$1 {
  constructor() {
    super(...arguments);
    /**
     * The index of the most recently checked option.
     *
     * @internal
     * @remarks
     * Multiple-selection mode only.
     */

    this.activeIndex = -1;
    /**
     * The start index when checking a range of options.
     *
     * @internal
     */

    this.rangeStartIndex = -1;
  }
  /**
   * Returns the last checked option.
   *
   * @internal
   */


  get activeOption() {
    return this.options[this.activeIndex];
  }
  /**
   * Returns the list of checked options.
   *
   * @internal
   */


  get checkedOptions() {
    var _a;

    return (_a = this.options) === null || _a === void 0 ? void 0 : _a.filter(o => o.checked);
  }
  /**
   * Returns the index of the first selected option.
   *
   * @internal
   */


  get firstSelectedOptionIndex() {
    return this.options.indexOf(this.firstSelectedOption);
  }
  /**
   * Updates the `ariaActiveDescendant` property when the active index changes.
   *
   * @param prev - the previous active index
   * @param next - the next active index
   *
   * @internal
   */


  activeIndexChanged(prev, next) {
    var _a, _b;

    this.ariaActiveDescendant = (_b = (_a = this.options[next]) === null || _a === void 0 ? void 0 : _a.id) !== null && _b !== void 0 ? _b : "";
    this.focusAndScrollOptionIntoView();
  }
  /**
   * Toggles the checked state for the currently active option.
   *
   * @remarks
   * Multiple-selection mode only.
   *
   * @internal
   */


  checkActiveIndex() {
    if (!this.multiple) {
      return;
    }

    const activeItem = this.activeOption;

    if (activeItem) {
      activeItem.checked = true;
    }
  }
  /**
   * Sets the active index to the first option and marks it as checked.
   *
   * @remarks
   * Multi-selection mode only.
   *
   * @param preserveChecked - mark all options unchecked before changing the active index
   *
   * @internal
   */


  checkFirstOption(preserveChecked = false) {
    if (preserveChecked) {
      if (this.rangeStartIndex === -1) {
        this.rangeStartIndex = this.activeIndex + 1;
      }

      this.options.forEach((o, i) => {
        o.checked = inRange(i, this.rangeStartIndex);
      });
    } else {
      this.uncheckAllOptions();
    }

    this.activeIndex = 0;
    this.checkActiveIndex();
  }
  /**
   * Decrements the active index and sets the matching option as checked.
   *
   * @remarks
   * Multi-selection mode only.
   *
   * @param preserveChecked - mark all options unchecked before changing the active index
   *
   * @internal
   */


  checkLastOption(preserveChecked = false) {
    if (preserveChecked) {
      if (this.rangeStartIndex === -1) {
        this.rangeStartIndex = this.activeIndex;
      }

      this.options.forEach((o, i) => {
        o.checked = inRange(i, this.rangeStartIndex, this.options.length);
      });
    } else {
      this.uncheckAllOptions();
    }

    this.activeIndex = this.options.length - 1;
    this.checkActiveIndex();
  }
  /**
   * @override
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.addEventListener("focusout", this.focusoutHandler);
  }
  /**
   * @override
   * @internal
   */


  disconnectedCallback() {
    this.removeEventListener("focusout", this.focusoutHandler);
    super.disconnectedCallback();
  }
  /**
   * Increments the active index and marks the matching option as checked.
   *
   * @remarks
   * Multiple-selection mode only.
   *
   * @param preserveChecked - mark all options unchecked before changing the active index
   *
   * @internal
   */


  checkNextOption(preserveChecked = false) {
    if (preserveChecked) {
      if (this.rangeStartIndex === -1) {
        this.rangeStartIndex = this.activeIndex;
      }

      this.options.forEach((o, i) => {
        o.checked = inRange(i, this.rangeStartIndex, this.activeIndex + 1);
      });
    } else {
      this.uncheckAllOptions();
    }

    this.activeIndex += this.activeIndex < this.options.length - 1 ? 1 : 0;
    this.checkActiveIndex();
  }
  /**
   * Decrements the active index and marks the matching option as checked.
   *
   * @remarks
   * Multiple-selection mode only.
   *
   * @param preserveChecked - mark all options unchecked before changing the active index
   *
   * @internal
   */


  checkPreviousOption(preserveChecked = false) {
    if (preserveChecked) {
      if (this.rangeStartIndex === -1) {
        this.rangeStartIndex = this.activeIndex;
      }

      if (this.checkedOptions.length === 1) {
        this.rangeStartIndex += 1;
      }

      this.options.forEach((o, i) => {
        o.checked = inRange(i, this.activeIndex, this.rangeStartIndex);
      });
    } else {
      this.uncheckAllOptions();
    }

    this.activeIndex -= this.activeIndex > 0 ? 1 : 0;
    this.checkActiveIndex();
  }
  /**
   * Handles click events for listbox options.
   *
   * @param e - the event object
   *
   * @override
   * @internal
   */


  clickHandler(e) {
    var _a;

    if (!this.multiple) {
      return super.clickHandler(e);
    }

    const captured = (_a = e.target) === null || _a === void 0 ? void 0 : _a.closest(`[role=option]`);

    if (!captured || captured.disabled) {
      return;
    }

    this.uncheckAllOptions();
    this.activeIndex = this.options.indexOf(captured);
    this.checkActiveIndex();
    this.toggleSelectedForAllCheckedOptions();
    return true;
  }
  /**
   * @override
   * @internal
   */


  focusAndScrollOptionIntoView() {
    super.focusAndScrollOptionIntoView(this.activeOption);
  }
  /**
   * In multiple-selection mode:
   * If any options are selected, the first selected option is checked when
   * the listbox receives focus. If no options are selected, the first
   * selectable option is checked.
   *
   * @override
   * @internal
   */


  focusinHandler(e) {
    if (!this.multiple) {
      return super.focusinHandler(e);
    }

    if (!this.shouldSkipFocus && e.target === e.currentTarget) {
      this.uncheckAllOptions();

      if (this.activeIndex === -1) {
        this.activeIndex = this.firstSelectedOptionIndex !== -1 ? this.firstSelectedOptionIndex : 0;
      }

      this.checkActiveIndex();
      this.setSelectedOptions();
      this.focusAndScrollOptionIntoView();
    }

    this.shouldSkipFocus = false;
  }
  /**
   * Unchecks all options when the listbox loses focus.
   *
   * @internal
   */


  focusoutHandler(e) {
    if (this.multiple) {
      this.uncheckAllOptions();
    }
  }
  /**
   * Handles keydown actions for listbox navigation and typeahead
   *
   * @override
   * @internal
   */


  keydownHandler(e) {
    if (!this.multiple) {
      return super.keydownHandler(e);
    }

    if (this.disabled) {
      return true;
    }

    const {
      key,
      shiftKey
    } = e;
    this.shouldSkipFocus = false;

    switch (key) {
      // Select the first available option
      case keyHome:
        {
          this.checkFirstOption(shiftKey);
          return;
        }
      // Select the next selectable option

      case keyArrowDown:
        {
          this.checkNextOption(shiftKey);
          return;
        }
      // Select the previous selectable option

      case keyArrowUp:
        {
          this.checkPreviousOption(shiftKey);
          return;
        }
      // Select the last available option

      case keyEnd:
        {
          this.checkLastOption(shiftKey);
          return;
        }

      case keyTab:
        {
          this.focusAndScrollOptionIntoView();
          return true;
        }

      case keyEscape:
        {
          this.uncheckAllOptions();
          this.checkActiveIndex();
          return true;
        }

      case keySpace:
        {
          e.preventDefault();

          if (this.typeAheadExpired) {
            this.toggleSelectedForAllCheckedOptions();
            return;
          }
        }
      // Send key to Typeahead handler

      default:
        {
          if (key.length === 1) {
            this.handleTypeAhead(`${key}`);
          }

          return true;
        }
    }
  }
  /**
   * Prevents `focusin` events from firing before `click` events when the
   * element is unfocused.
   *
   * @override
   * @internal
   */


  mousedownHandler(e) {
    if (e.offsetX >= 0 && e.offsetX <= this.scrollWidth) {
      return super.mousedownHandler(e);
    }
  }
  /**
   * Switches between single-selection and multi-selection mode.
   *
   * @internal
   */


  multipleChanged(prev, next) {
    var _a;

    this.ariaMultiSelectable = next ? "true" : null;
    (_a = this.options) === null || _a === void 0 ? void 0 : _a.forEach(o => {
      o.checked = next ? false : undefined;
    });
    this.setSelectedOptions();
  }
  /**
   * Sets an option as selected and gives it focus.
   *
   * @override
   * @public
   */


  setSelectedOptions() {
    if (!this.multiple) {
      super.setSelectedOptions();
      return;
    }

    if (this.$fastController.isConnected && this.options) {
      this.selectedOptions = this.options.filter(o => o.selected);
      this.focusAndScrollOptionIntoView();
    }
  }
  /**
   * Ensures the size is a positive integer when the property is updated.
   *
   * @param prev - the previous size value
   * @param next - the current size value
   *
   * @internal
   */


  sizeChanged(prev, next) {
    var _a;

    const size = Math.max(0, parseInt((_a = next === null || next === void 0 ? void 0 : next.toFixed()) !== null && _a !== void 0 ? _a : "", 10));

    if (size !== next) {
      DOM.queueUpdate(() => {
        this.size = size;
      });
    }
  }
  /**
   * Toggles the selected state of the provided options. If any provided items
   * are in an unselected state, all items are set to selected. If every
   * provided item is selected, they are all unselected.
   *
   * @internal
   */


  toggleSelectedForAllCheckedOptions() {
    const enabledCheckedOptions = this.checkedOptions.filter(o => !o.disabled);
    const force = !enabledCheckedOptions.every(o => o.selected);
    enabledCheckedOptions.forEach(o => o.selected = force);
    this.selectedIndex = this.options.indexOf(enabledCheckedOptions[enabledCheckedOptions.length - 1]);
    this.setSelectedOptions();
  }
  /**
   * @override
   * @internal
   */


  typeaheadBufferChanged(prev, next) {
    if (!this.multiple) {
      super.typeaheadBufferChanged(prev, next);
      return;
    }

    if (this.$fastController.isConnected) {
      const typeaheadMatches = this.getTypeaheadMatches();
      const activeIndex = this.options.indexOf(typeaheadMatches[0]);

      if (activeIndex > -1) {
        this.activeIndex = activeIndex;
        this.uncheckAllOptions();
        this.checkActiveIndex();
      }

      this.typeAheadExpired = false;
    }
  }
  /**
   * Unchecks all options.
   *
   * @remarks
   * Multiple-selection mode only.
   *
   * @param preserveChecked - reset the rangeStartIndex
   *
   * @internal
   */


  uncheckAllOptions(preserveChecked = false) {
    this.options.forEach(o => o.checked = this.multiple ? false : undefined);

    if (!preserveChecked) {
      this.rangeStartIndex = -1;
    }
  }

}

__decorate$1([observable], ListboxElement.prototype, "activeIndex", void 0);

__decorate$1([attr({
  mode: "boolean"
})], ListboxElement.prototype, "multiple", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], ListboxElement.prototype, "size", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(Listbox:class)} component.
 * @public
 */

const listboxTemplate = (context, definition) => html`<template aria-activedescendant="${x => x.ariaActiveDescendant}" aria-multiselectable="${x => x.ariaMultiSelectable}" class="listbox" role="listbox" tabindex="${x => !x.disabled ? "0" : null}" @click="${(x, c) => x.clickHandler(c.event)}" @focusin="${(x, c) => x.focusinHandler(c.event)}" @keydown="${(x, c) => x.keydownHandler(c.event)}" @mousedown="${(x, c) => x.mousedownHandler(c.event)}"><slot ${slotted({
  filter: ListboxElement.slottedOptionFilter,
  flatten: true,
  property: "slottedOptions"
})}></slot></template>`;

/**
 * A List Picker Menu Custom HTML Element.
 *
 * @alpha
 */

class PickerMenu$1 extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * Children that are list items
     *
     * @internal
     */

    this.optionElements = [];
  }

  menuElementsChanged() {
    this.updateOptions();
  }

  headerElementsChanged() {
    this.updateOptions();
  }

  footerElementsChanged() {
    this.updateOptions();
  }

  updateOptions() {
    this.optionElements.splice(0, this.optionElements.length);
    this.addSlottedListItems(this.headerElements);
    this.addSlottedListItems(this.menuElements);
    this.addSlottedListItems(this.footerElements);
    this.$emit("optionsupdated", {
      bubbles: false
    });
  }

  addSlottedListItems(slotChildren) {
    if (slotChildren === undefined) {
      return;
    }

    slotChildren.forEach(child => {
      if (child.nodeType === 1 && child.getAttribute("role") === "listitem") {
        child.id = child.id || uniqueId("option-");
        this.optionElements.push(child);
      }
    });
  }

}

__decorate$1([observable], PickerMenu$1.prototype, "menuElements", void 0);

__decorate$1([observable], PickerMenu$1.prototype, "headerElements", void 0);

__decorate$1([observable], PickerMenu$1.prototype, "footerElements", void 0);

__decorate$1([observable], PickerMenu$1.prototype, "suggestionsAvailableText", void 0);

const defaultContentsTemplate$1 = html`<template>${x => x.value}</template>`;
/**
 * A picker list item Custom HTML Element.
 *
 * @alpha
 */

class PickerMenuOption extends FoundationElement {
  contentsTemplateChanged() {
    if (this.$fastController.isConnected) {
      this.updateView();
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.updateView();
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    super.disconnectedCallback();
    this.disconnectView();
  }

  handleClick(e) {
    if (e.defaultPrevented) {
      return false;
    }

    this.handleInvoked();
    return false;
  }

  handleInvoked() {
    this.$emit("pickeroptioninvoked");
  }

  updateView() {
    var _a, _b;

    this.disconnectView();
    this.customView = (_b = (_a = this.contentsTemplate) === null || _a === void 0 ? void 0 : _a.render(this, this)) !== null && _b !== void 0 ? _b : defaultContentsTemplate$1.render(this, this);
  }

  disconnectView() {
    var _a;

    (_a = this.customView) === null || _a === void 0 ? void 0 : _a.dispose();
    this.customView = undefined;
  }

}

__decorate$1([attr({
  attribute: "value"
})], PickerMenuOption.prototype, "value", void 0);

__decorate$1([observable], PickerMenuOption.prototype, "contentsTemplate", void 0);

/**
 * A List Picker Menu Custom HTML Element.
 *
 * @alpha
 */

class PickerList extends FoundationElement {}

const defaultContentsTemplate = html`<template>${x => x.value}</template>`;
/**
 * A picker list item Custom HTML Element.
 *
 * @alpha
 */

class PickerListItem extends FoundationElement {
  contentsTemplateChanged() {
    if (this.$fastController.isConnected) {
      this.updateView();
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.updateView();
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    this.disconnectView();
    super.disconnectedCallback();
  }

  handleKeyDown(e) {
    if (e.defaultPrevented) {
      return false;
    }

    if (e.key === keyEnter) {
      this.handleInvoke();
      return false;
    }

    return true;
  }

  handleClick(e) {
    if (!e.defaultPrevented) {
      this.handleInvoke();
    }

    return false;
  }

  handleInvoke() {
    this.$emit("pickeriteminvoked");
  }

  updateView() {
    var _a, _b;

    this.disconnectView();
    this.customView = (_b = (_a = this.contentsTemplate) === null || _a === void 0 ? void 0 : _a.render(this, this)) !== null && _b !== void 0 ? _b : defaultContentsTemplate.render(this, this);
  }

  disconnectView() {
    var _a;

    (_a = this.customView) === null || _a === void 0 ? void 0 : _a.dispose();
    this.customView = undefined;
  }

}

__decorate$1([attr({
  attribute: "value"
})], PickerListItem.prototype, "value", void 0);

__decorate$1([observable], PickerListItem.prototype, "contentsTemplate", void 0);

function createDefaultListItemTemplate(context) {
  const pickerListItemTag = context.tagFor(PickerListItem);
  return html`<${pickerListItemTag} value="${x => x}" :contentsTemplate="${(x, c) => c.parent.listItemContentsTemplate}"></${pickerListItemTag}>`;
}

function createDefaultMenuOptionTemplate(context) {
  const pickerMenuOptionTag = context.tagFor(PickerMenuOption);
  return html`<${pickerMenuOptionTag} value="${x => x}" :contentsTemplate="${(x, c) => c.parent.menuOptionContentsTemplate}"></${pickerMenuOptionTag}>`;
}
/**
 * The template for the List Picker component.
 * @public
 */


const pickerTemplate = (context, definition) => {
  const anchoredRegionTag = context.tagFor(AnchoredRegion);
  const pickerMenuTag = context.tagFor(PickerMenu$1);
  const pickerListTag = context.tagFor(PickerList);
  const progressRingTag = context.tagFor(PickerList);
  const defaultListItemTemplate = createDefaultListItemTemplate(context);
  const defaultMenuOptionTemplate = createDefaultMenuOptionTemplate(context);
  return html`<template :selectedListTag="${() => pickerListTag}" :menuTag="${() => pickerMenuTag}" :defaultListItemTemplate="${defaultListItemTemplate}" :defaultMenuOptionTemplate="${defaultMenuOptionTemplate}" @focusin="${(x, c) => x.handleFocusIn(c.event)}" @focusout="${(x, c) => x.handleFocusOut(c.event)}" @keydown="${(x, c) => x.handleKeyDown(c.event)}" @pickeriteminvoked="${(x, c) => x.handleItemInvoke(c.event)}" @pickeroptioninvoked="${(x, c) => x.handleOptionInvoke(c.event)}"><slot name="list-region"></slot>${when(x => x.flyoutOpen, html`<${anchoredRegionTag} class="region" part="region" auto-update-mode="${x => x.menuConfig.autoUpdateMode}" fixed-placement="${x => x.menuConfig.fixedPlacement}" vertical-positioning-mode="${x => x.menuConfig.verticalPositioningMode}" vertical-default-position="${x => x.menuConfig.verticalDefaultPosition}" vertical-scaling="${x => x.menuConfig.verticalScaling}" vertical-inset="${x => x.menuConfig.verticalInset}" vertical-viewport-lock="${x => x.menuConfig.verticalViewportLock}" horizontal-positioning-mode="${x => x.menuConfig.horizontalPositioningMode}" horizontal-default-position="${x => x.menuConfig.horizontalDefaultPosition}" horizontal-scaling="${x => x.menuConfig.horizontalScaling}" horizontal-inset="${x => x.menuConfig.horizontalInset}" horizontal-viewport-lock="${x => x.menuConfig.horizontalViewportLock}" @loaded="${(x, c) => x.handleRegionLoaded(c.event)}" ${ref("region")}>${when(x => !x.showNoOptions && !x.showLoading, html`<slot name="menu-region"></slot>`)} ${when(x => x.showNoOptions && !x.showLoading, html`<div class="no-options-display" part="no-options-display"><slot name="no-options-region">${x => x.noSuggestionsText}</slot></div>`)} ${when(x => x.showLoading, html`<div class="loading-display" part="loading-display"><slot name="loading-region"><${progressRingTag} part="loading-progress" class="loading-progress slot="loading-region"></${progressRingTag}>${x => x.loadingText}</slot></div>`)}</${anchoredRegionTag}>`)}</template>`;
};

class _Picker extends FoundationElement {}
/**
 * A form-associated base class for the {@link @microsoft/fast-foundation#(Picker:class)} component.
 *
 * @internal
 */


class FormAssociatedPicker extends FormAssociated(_Picker) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("input");
  }

}

const pickerInputTemplate = html`<input slot="input-region" role="combobox" type="text" autocapitalize="off" autocomplete="off" haspopup="list" aria-label="${x => x.label}" aria-labelledby="${x => x.labelledBy}" placeholder="${x => x.placeholder}" ${ref("inputElement")}></input>`;
/**
 * A Picker Custom HTML Element.  This is an early "alpha" version of the component.
 * Developers should expect the api to evolve, breaking changes are possible.
 *
 * @alpha
 */

class Picker extends FormAssociatedPicker {
  constructor() {
    super(...arguments);
    /**
     * Currently selected items. Comma delineated string ie. "apples,oranges".
     *
     * @alpha
     * @remarks
     * HTML Attribute: selection
     */

    this.selection = "";
    /**
     * Whether the component should remove an option from the list when it is in the selection
     *
     * @alpha
     * @remarks
     * HTML Attribute: filter-selected
     */

    this.filterSelected = true;
    /**
     * Whether the component should remove options based on the current query
     *
     * @alpha
     * @remarks
     * HTML Attribute: filter-query
     */

    this.filterQuery = true;
    /**
     * The text to present to assistive technolgies when no suggestions are available.
     *
     * @alpha
     * @remarks
     * HTML Attribute: no-suggestions-text
     */

    this.noSuggestionsText = "No suggestions available";
    /**
     *  The text to present to assistive technolgies when suggestions are available.
     *
     * @alpha
     * @remarks
     * HTML Attribute: suggestions-available-text
     */

    this.suggestionsAvailableText = "Suggestions available";
    /**
     * The text to present to assistive technologies when suggestions are loading.
     *
     * @alpha
     * @remarks
     * HTML Attribute: loading-text
     */

    this.loadingText = "Loading suggestions";
    /**
     * Controls menu placement
     *
     * @alpha
     * @remarks
     * HTML Attribute: menu-placement
     */

    this.menuPlacement = "bottom-fill";
    /**
     * Whether to display a loading state if the menu is opened.
     *
     * @alpha
     */

    this.showLoading = false;
    /**
     *  Current list of options in array form
     *
     * @alpha
     */

    this.optionsList = [];
    /**
     *  Current list of filtered options in array form
     *
     * @internal
     */

    this.filteredOptionsList = [];
    /**
     *  Indicates if the flyout menu is open or not
     *
     * @internal
     */

    this.flyoutOpen = false;
    /**
     *  Index of currently active menu option
     *
     * @internal
     */

    this.menuFocusIndex = -1;
    /**
     *  Internal flag to indicate no options available display should be shown.
     *
     * @internal
     */

    this.showNoOptions = false;
    /**
     *
     *
     * @internal
     */

    this.selectedItems = [];
    this.inputElementView = null;
    /**
     * Handle input event from input element
     */

    this.handleTextInput = e => {
      this.query = this.inputElement.value;
    };
    /**
     * Handle click event from input element
     */


    this.handleInputClick = e => {
      e.preventDefault();
      this.toggleFlyout(true);
    };
    /**
     * Sets properties on the anchored region once it is instanciated.
     */


    this.setRegionProps = () => {
      if (!this.flyoutOpen) {
        return;
      }

      if (this.region === null || this.region === undefined) {
        // TODO: limit this
        DOM.queueUpdate(this.setRegionProps);
        return;
      }

      this.region.anchorElement = this.inputElement;
    };
    /**
     * matches menu placement values with the associated menu config
     */


    this.configLookup = {
      top: FlyoutPosTop,
      bottom: FlyoutPosBottom,
      tallest: FlyoutPosTallest,
      "top-fill": FlyoutPosTopFill,
      "bottom-fill": FlyoutPosBottomFill,
      "tallest-fill": FlyoutPosTallestFill
    };
  }

  selectionChanged() {
    if (this.$fastController.isConnected) {
      this.handleSelectionChange();

      if (this.proxy instanceof HTMLInputElement) {
        this.proxy.value = this.selection;
        this.validate();
      }
    }
  }

  optionsChanged() {
    this.optionsList = this.options.split(",").map(opt => opt.trim()).filter(opt => opt !== "");
  }

  menuPlacementChanged() {
    if (this.$fastController.isConnected) {
      this.updateMenuConfig();
    }
  }

  showLoadingChanged() {
    if (this.$fastController.isConnected) {
      DOM.queueUpdate(() => {
        this.setFocusedOption(0);
      });
    }
  }

  listItemTemplateChanged() {
    this.updateListItemTemplate();
  }

  defaultListItemTemplateChanged() {
    this.updateListItemTemplate();
  }

  menuOptionTemplateChanged() {
    this.updateOptionTemplate();
  }

  defaultMenuOptionTemplateChanged() {
    this.updateOptionTemplate();
  }

  optionsListChanged() {
    this.updateFilteredOptions();
  }

  queryChanged() {
    if (this.$fastController.isConnected) {
      if (this.inputElement.value !== this.query) {
        this.inputElement.value = this.query;
      }

      this.updateFilteredOptions();
      this.$emit("querychange", {
        bubbles: false
      });
    }
  }

  filteredOptionsListChanged() {
    if (this.$fastController.isConnected) {
      this.showNoOptions = this.filteredOptionsList.length === 0 && this.menuElement.querySelectorAll('[role="listitem"]').length === 0;
      this.setFocusedOption(this.showNoOptions ? -1 : 0);
    }
  }

  flyoutOpenChanged() {
    if (this.flyoutOpen) {
      DOM.queueUpdate(this.setRegionProps);
      this.$emit("menuopening", {
        bubbles: false
      });
    } else {
      this.$emit("menuclosing", {
        bubbles: false
      });
    }
  }

  showNoOptionsChanged() {
    if (this.$fastController.isConnected) {
      DOM.queueUpdate(() => {
        this.setFocusedOption(0);
      });
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.listElement = document.createElement(this.selectedListTag);
    this.appendChild(this.listElement);
    this.itemsPlaceholderElement = document.createComment("");
    this.listElement.append(this.itemsPlaceholderElement);
    this.inputElementView = pickerInputTemplate.render(this, this.listElement);
    const match = this.menuTag.toUpperCase();
    this.menuElement = Array.from(this.children).find(element => {
      return element.tagName === match;
    });

    if (this.menuElement === undefined) {
      this.menuElement = document.createElement(this.menuTag);
      this.appendChild(this.menuElement);
    }

    if (this.menuElement.id === "") {
      this.menuElement.id = uniqueId("listbox-");
    }

    this.menuId = this.menuElement.id;
    this.optionsPlaceholder = document.createComment("");
    this.menuElement.append(this.optionsPlaceholder);
    this.updateMenuConfig();
    DOM.queueUpdate(() => this.initialize());
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.toggleFlyout(false);
    this.inputElement.removeEventListener("input", this.handleTextInput);
    this.inputElement.removeEventListener("click", this.handleInputClick);

    if (this.inputElementView !== null) {
      this.inputElementView.dispose();
      this.inputElementView = null;
    }
  }
  /**
   * Move focus to the input element
   * @public
   */


  focus() {
    this.inputElement.focus();
  }
  /**
   * Initialize the component.  This is delayed a frame to ensure children are connected as well.
   */


  initialize() {
    this.updateListItemTemplate();
    this.updateOptionTemplate();
    this.itemsRepeatBehavior = new RepeatDirective(x => x.selectedItems, x => x.activeListItemTemplate, {
      positioning: true
    }).createBehavior(this.itemsPlaceholderElement);
    this.inputElement.addEventListener("input", this.handleTextInput);
    this.inputElement.addEventListener("click", this.handleInputClick);
    /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */

    this.$fastController.addBehaviors([this.itemsRepeatBehavior]);
    this.menuElement.suggestionsAvailableText = this.suggestionsAvailableText;
    this.menuElement.addEventListener("optionsupdated", this.handleMenuOptionsUpdated);
    this.optionsRepeatBehavior = new RepeatDirective(x => x.filteredOptionsList, x => x.activeMenuOptionTemplate, {
      positioning: true
    }).createBehavior(this.optionsPlaceholder);
    /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */

    this.$fastController.addBehaviors([this.optionsRepeatBehavior]);
    this.handleSelectionChange();
  }
  /**
   * Toggles the menu flyout
   */


  toggleFlyout(open) {
    if (this.flyoutOpen === open) {
      return;
    }

    if (open && document.activeElement === this.inputElement) {
      this.flyoutOpen = open;
      DOM.queueUpdate(() => {
        if (this.menuElement !== undefined) {
          this.setFocusedOption(0);
        } else {
          this.disableMenu();
        }
      });
      return;
    }

    this.flyoutOpen = false;
    this.disableMenu();
    return;
  }
  /**
   * Handle the menu options updated event from the child menu
   */


  handleMenuOptionsUpdated(e) {
    e.preventDefault();

    if (this.flyoutOpen) {
      this.setFocusedOption(0);
    }
  }
  /**
   * Handle key down events.
   */


  handleKeyDown(e) {
    if (e.defaultPrevented) {
      return false;
    }

    switch (e.key) {
      // TODO: what should "home" and "end" keys do, exactly?
      //
      // case keyHome: {
      //     if (!this.flyoutOpen) {
      //         this.toggleFlyout(true);
      //     } else {
      //         if (this.menuElement.optionElements.length > 0) {
      //             this.setFocusedOption(0);
      //         }
      //     }
      //     return false;
      // }
      // case keyEnd: {
      //     if (!this.flyoutOpen) {
      //         this.toggleFlyout(true);
      //     } else {
      //         if (this.menuElement.optionElements.length > 0) {
      //             this.toggleFlyout(true);
      //             this.setFocusedOption(this.menuElement.optionElements.length - 1);
      //         }
      //     }
      //     return false;
      // }
      case keyArrowDown:
        {
          if (!this.flyoutOpen) {
            this.toggleFlyout(true);
          } else {
            const nextFocusOptionIndex = this.flyoutOpen ? Math.min(this.menuFocusIndex + 1, this.menuElement.optionElements.length - 1) : 0;
            this.setFocusedOption(nextFocusOptionIndex);
          }

          return false;
        }

      case keyArrowUp:
        {
          if (!this.flyoutOpen) {
            this.toggleFlyout(true);
          } else {
            const previousFocusOptionIndex = this.flyoutOpen ? Math.max(this.menuFocusIndex - 1, 0) : 0;
            this.setFocusedOption(previousFocusOptionIndex);
          }

          return false;
        }

      case keyEscape:
        {
          this.toggleFlyout(false);
          return false;
        }

      case keyEnter:
        {
          if (this.menuFocusIndex !== -1 && this.menuElement.optionElements.length > this.menuFocusIndex) {
            this.menuElement.optionElements[this.menuFocusIndex].click();
          }

          return false;
        }

      case keyArrowRight:
        {
          if (document.activeElement !== this.inputElement) {
            this.incrementFocusedItem(1);
            return false;
          } // don't block if arrow keys moving caret in input element


          return true;
        }

      case keyArrowLeft:
        {
          if (this.inputElement.selectionStart === 0) {
            this.incrementFocusedItem(-1);
            return false;
          } // don't block if arrow keys moving caret in input element


          return true;
        }

      case keyDelete:
      case keyBackspace:
        {
          if (document.activeElement === null) {
            return true;
          }

          if (document.activeElement === this.inputElement) {
            if (this.inputElement.selectionStart === 0) {
              this.selection = this.selectedItems.slice(0, this.selectedItems.length - 1).toString();
              this.toggleFlyout(false);
              return false;
            } // let text deletion proceed


            return true;
          }

          const selectedItems = Array.from(this.listElement.children);
          const currentFocusedItemIndex = selectedItems.indexOf(document.activeElement);

          if (currentFocusedItemIndex > -1) {
            // delete currently focused item
            this.selection = this.selectedItems.splice(currentFocusedItemIndex, 1).toString();
            DOM.queueUpdate(() => {
              selectedItems[Math.min(selectedItems.length, currentFocusedItemIndex)].focus();
            });
            return false;
          }

          return true;
        }
    }

    this.toggleFlyout(true);
    return true;
  }
  /**
   * Handle focus in events.
   */


  handleFocusIn(e) {
    return false;
  }
  /**
   * Handle focus out events.
   */


  handleFocusOut(e) {
    if (this.menuElement === undefined || !this.menuElement.contains(e.relatedTarget)) {
      this.toggleFlyout(false);
    }

    return false;
  }
  /**
   * The list of selected items has changed
   */


  handleSelectionChange() {
    if (this.selectedItems.toString() === this.selection) {
      return;
    }

    this.selectedItems = this.selection === "" ? [] : this.selection.split(",");
    this.updateFilteredOptions();
    DOM.queueUpdate(() => {
      this.checkMaxItems();
    });
    this.$emit("selectionchange", {
      bubbles: false
    });
  }
  /**
   * Anchored region is loaded, menu and options exist in the DOM.
   */


  handleRegionLoaded(e) {
    DOM.queueUpdate(() => {
      this.setFocusedOption(0);
      this.$emit("menuloaded", {
        bubbles: false
      });
    });
  }
  /**
   * Checks if the maximum number of items has been chosen and updates the ui.
   */


  checkMaxItems() {
    if (this.inputElement === undefined) {
      return;
    }

    if (this.maxSelected !== undefined && this.selectedItems.length >= this.maxSelected) {
      if (document.activeElement === this.inputElement) {
        const selectedItemInstances = Array.from(this.listElement.querySelectorAll("[role='listitem']"));
        selectedItemInstances[selectedItemInstances.length - 1].focus();
      }

      this.inputElement.hidden = true;
    } else {
      this.inputElement.hidden = false;
    }
  }
  /**
   * A list item has been invoked.
   */


  handleItemInvoke(e) {
    if (e.defaultPrevented) {
      return false;
    }

    if (e.target instanceof PickerListItem) {
      const listItems = Array.from(this.listElement.querySelectorAll("[role='listitem']"));
      const itemIndex = listItems.indexOf(e.target);

      if (itemIndex !== -1) {
        const newSelection = this.selectedItems.slice();
        newSelection.splice(itemIndex, 1);
        this.selection = newSelection.toString();
        DOM.queueUpdate(() => this.incrementFocusedItem(0));
      }

      return false;
    }

    return true;
  }
  /**
   * A menu option has been invoked.
   */


  handleOptionInvoke(e) {
    if (e.defaultPrevented) {
      return false;
    }

    if (e.target instanceof PickerMenuOption) {
      if (e.target.value !== undefined) {
        this.selection = `${this.selection}${this.selection === "" ? "" : ","}${e.target.value}`;
      }

      this.inputElement.value = "";
      this.query = "";
      this.inputElement.focus();
      this.toggleFlyout(false);
      return false;
    } // const value: string = (e.target as PickerMenuOption).value;


    return true;
  }
  /**
   * Increments the focused list item by the specified amount
   */


  incrementFocusedItem(increment) {
    if (this.selectedItems.length === 0) {
      this.inputElement.focus();
      return;
    }

    const selectedItemsAsElements = Array.from(this.listElement.querySelectorAll("[role='listitem']"));

    if (document.activeElement !== null) {
      let currentFocusedItemIndex = selectedItemsAsElements.indexOf(document.activeElement);

      if (currentFocusedItemIndex === -1) {
        // use the input element
        currentFocusedItemIndex = selectedItemsAsElements.length;
      }

      const newFocusedItemIndex = Math.min(selectedItemsAsElements.length, Math.max(0, currentFocusedItemIndex + increment));

      if (newFocusedItemIndex === selectedItemsAsElements.length) {
        if (this.maxSelected !== undefined && this.selectedItems.length >= this.maxSelected) {
          selectedItemsAsElements[newFocusedItemIndex - 1].focus();
        } else {
          this.inputElement.focus();
        }
      } else {
        selectedItemsAsElements[newFocusedItemIndex].focus();
      }
    }
  }
  /**
   * Disables the menu. Note that the menu can be open, just doens't have any valid options on display.
   */


  disableMenu() {
    var _a, _b, _c;

    this.menuFocusIndex = -1;
    this.menuFocusOptionId = undefined;
    (_a = this.inputElement) === null || _a === void 0 ? void 0 : _a.removeAttribute("aria-activedescendant");
    (_b = this.inputElement) === null || _b === void 0 ? void 0 : _b.removeAttribute("aria-owns");
    (_c = this.inputElement) === null || _c === void 0 ? void 0 : _c.removeAttribute("aria-expanded");
  }
  /**
   * Sets the currently focused menu option by index
   */


  setFocusedOption(optionIndex) {
    if (!this.flyoutOpen || optionIndex === -1 || this.showNoOptions || this.showLoading) {
      this.disableMenu();
      return;
    }

    if (this.menuElement.optionElements.length === 0) {
      return;
    }

    this.menuElement.optionElements.forEach(element => {
      element.setAttribute("aria-selected", "false");
    });
    this.menuFocusIndex = optionIndex;

    if (this.menuFocusIndex > this.menuElement.optionElements.length - 1) {
      this.menuFocusIndex = this.menuElement.optionElements.length - 1;
    }

    this.menuFocusOptionId = this.menuElement.optionElements[this.menuFocusIndex].id;
    this.inputElement.setAttribute("aria-owns", this.menuId);
    this.inputElement.setAttribute("aria-expanded", "true");
    this.inputElement.setAttribute("aria-activedescendant", this.menuFocusOptionId);
    const focusedOption = this.menuElement.optionElements[this.menuFocusIndex];
    focusedOption.setAttribute("aria-selected", "true");
    this.menuElement.scrollTo(0, focusedOption.offsetTop);
  }
  /**
   * Updates the template used for the list item repeat behavior
   */


  updateListItemTemplate() {
    var _a;

    this.activeListItemTemplate = (_a = this.listItemTemplate) !== null && _a !== void 0 ? _a : this.defaultListItemTemplate;
  }
  /**
   * Updates the template used for the menu option repeat behavior
   */


  updateOptionTemplate() {
    var _a;

    this.activeMenuOptionTemplate = (_a = this.menuOptionTemplate) !== null && _a !== void 0 ? _a : this.defaultMenuOptionTemplate;
  }
  /**
   * Updates the filtered options array
   */


  updateFilteredOptions() {
    this.filteredOptionsList = this.optionsList.slice(0);

    if (this.filterSelected) {
      this.filteredOptionsList = this.filteredOptionsList.filter(el => this.selectedItems.indexOf(el) === -1);
    }

    if (this.filterQuery && this.query !== "" && this.query !== undefined) {
      this.filteredOptionsList = this.filteredOptionsList.filter(el => el.indexOf(this.query) !== -1);
    }
  }
  /**
   * Updates the menu configuration
   */


  updateMenuConfig() {
    let newConfig = this.configLookup[this.menuPlacement];

    if (newConfig === null) {
      newConfig = FlyoutPosBottomFill;
    }

    this.menuConfig = Object.assign(Object.assign({}, newConfig), {
      autoUpdateMode: "auto",
      fixedPlacement: true,
      horizontalViewportLock: false,
      verticalViewportLock: false
    });
  }

}

__decorate$1([attr({
  attribute: "selection"
})], Picker.prototype, "selection", void 0);

__decorate$1([attr({
  attribute: "options"
})], Picker.prototype, "options", void 0);

__decorate$1([attr({
  attribute: "filter-selected",
  mode: "boolean"
})], Picker.prototype, "filterSelected", void 0);

__decorate$1([attr({
  attribute: "filter-query",
  mode: "boolean"
})], Picker.prototype, "filterQuery", void 0);

__decorate$1([attr({
  attribute: "max-selected"
})], Picker.prototype, "maxSelected", void 0);

__decorate$1([attr({
  attribute: "no-suggestions-text"
})], Picker.prototype, "noSuggestionsText", void 0);

__decorate$1([attr({
  attribute: "suggestions-available-text"
})], Picker.prototype, "suggestionsAvailableText", void 0);

__decorate$1([attr({
  attribute: "loading-text"
})], Picker.prototype, "loadingText", void 0);

__decorate$1([attr({
  attribute: "label"
})], Picker.prototype, "label", void 0);

__decorate$1([attr({
  attribute: "labelledby"
})], Picker.prototype, "labelledBy", void 0);

__decorate$1([attr({
  attribute: "placeholder"
})], Picker.prototype, "placeholder", void 0);

__decorate$1([attr({
  attribute: "menu-placement"
})], Picker.prototype, "menuPlacement", void 0);

__decorate$1([observable], Picker.prototype, "showLoading", void 0);

__decorate$1([observable], Picker.prototype, "listItemTemplate", void 0);

__decorate$1([observable], Picker.prototype, "defaultListItemTemplate", void 0);

__decorate$1([observable], Picker.prototype, "activeListItemTemplate", void 0);

__decorate$1([observable], Picker.prototype, "menuOptionTemplate", void 0);

__decorate$1([observable], Picker.prototype, "defaultMenuOptionTemplate", void 0);

__decorate$1([observable], Picker.prototype, "activeMenuOptionTemplate", void 0);

__decorate$1([observable], Picker.prototype, "listItemContentsTemplate", void 0);

__decorate$1([observable], Picker.prototype, "menuOptionContentsTemplate", void 0);

__decorate$1([observable], Picker.prototype, "optionsList", void 0);

__decorate$1([observable], Picker.prototype, "query", void 0);

__decorate$1([observable], Picker.prototype, "filteredOptionsList", void 0);

__decorate$1([observable], Picker.prototype, "flyoutOpen", void 0);

__decorate$1([observable], Picker.prototype, "menuId", void 0);

__decorate$1([observable], Picker.prototype, "selectedListTag", void 0);

__decorate$1([observable], Picker.prototype, "menuTag", void 0);

__decorate$1([observable], Picker.prototype, "menuFocusIndex", void 0);

__decorate$1([observable], Picker.prototype, "menuFocusOptionId", void 0);

__decorate$1([observable], Picker.prototype, "showNoOptions", void 0);

__decorate$1([observable], Picker.prototype, "menuConfig", void 0);

__decorate$1([observable], Picker.prototype, "selectedItems", void 0);

/**
 * The template for the List Picker component.
 * @public
 */

const pickerMenuTemplate = (context, definition) => {
  return html`<template role="list" slot="menu-region"><div class="options-display" part="options-display"><div class="header-region" part="header-region"><slot name="header-region" ${slotted("headerElements")}></slot></div><slot ${slotted("menuElements")}></slot><div class="footer-region" part="footer-region"><slot name="footer-region" ${slotted("footerElements")}></slot></div><div role="alert" aria-live="polite" part="suggestions-available-alert" class="suggestions-available-alert">${x => x.suggestionsAvailableText}</div></div></template>`;
};

/**
 *
 * @public
 */

const pickerMenuOptionTemplate = (context, definition) => {
  return html`<template role="listitem" tabindex="-1" @click="${(x, c) => x.handleClick(c.event)}"><slot></slot></template>`;
};

/**
 *
 * @public
 */

const pickerListTemplate = (context, definition) => {
  return html`<template slot="list-region" role="list" class="picker-list"><slot></slot><slot name="input-region"></slot></template>`;
};

/**
 *
 * @public
 */

const pickerListItemTemplate = (context, definition) => {
  return html`<template role="listitem" tabindex="0" @click="${(x, c) => x.handleClick(c.event)}" @keydown="${(x, c) => x.handleKeyDown(c.event)}"><slot></slot></template>`;
};

/**
 * Menu items roles.
 * @public
 */
const MenuItemRole = {
  /**
   * The menu item has a "menuitem" role
   */
  menuitem: "menuitem",

  /**
   * The menu item has a "menuitemcheckbox" role
   */
  menuitemcheckbox: "menuitemcheckbox",

  /**
   * The menu item has a "menuitemradio" role
   */
  menuitemradio: "menuitemradio"
};
/**
 * @internal
 */

const roleForMenuItem = {
  [MenuItemRole.menuitem]: "menuitem",
  [MenuItemRole.menuitemcheckbox]: "menuitemcheckbox",
  [MenuItemRole.menuitemradio]: "menuitemradio"
};

/**
 * A Switch Custom HTML Element.
 * Implements {@link https://www.w3.org/TR/wai-aria-1.1/#menuitem | ARIA menuitem }, {@link https://www.w3.org/TR/wai-aria-1.1/#menuitemcheckbox | ARIA menuitemcheckbox}, or {@link https://www.w3.org/TR/wai-aria-1.1/#menuitemradio | ARIA menuitemradio }.
 *
 * @slot checked-indicator - The checked indicator
 * @slot radio-indicator - The radio indicator
 * @slot start - Content which can be provided before the menu item content
 * @slot end - Content which can be provided after the menu item content
 * @slot - The default slot for menu item content
 * @slot expand-collapse-indicator - The expand/collapse indicator
 * @slot submenu - Used to nest menu's within menu items
 * @csspart input-container - The element representing the visual checked or radio indicator
 * @csspart checkbox - The element wrapping the `menuitemcheckbox` indicator
 * @csspart radio - The element wrapping the `menuitemradio` indicator
 * @csspart content - The element wrapping the menu item content
 * @csspart expand-collapse-glyph-container - The element wrapping the expand collapse element
 * @csspart expand-collapse - The expand/collapse element
 * @csspart submenu-region - The container for the submenu, used for positioning
 * @fires expanded-change - Fires a custom 'expanded-change' event when the expanded state changes
 * @fires change - Fires a custom 'change' event when a non-submenu item with a role of `menuitemcheckbox`, `menuitemradio`, or `menuitem` is invoked
 *
 * @public
 */

class MenuItem extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * The role of the element.
     *
     * @public
     * @remarks
     * HTML Attribute: role
     */

    this.role = MenuItemRole.menuitem;
    /**
     * @internal
     */

    this.hasSubmenu = false;
    /**
     * Track current direction to pass to the anchored region
     *
     * @internal
     */

    this.currentDirection = Direction.ltr;
    this.focusSubmenuOnLoad = false;
    /**
     * @internal
     */

    this.handleMenuItemKeyDown = e => {
      if (e.defaultPrevented) {
        return false;
      }

      switch (e.key) {
        case keyEnter:
        case keySpace:
          this.invoke();
          return false;

        case keyArrowRight:
          //open/focus on submenu
          this.expandAndFocus();
          return false;

        case keyArrowLeft:
          //close submenu
          if (this.expanded) {
            this.expanded = false;
            this.focus();
            return false;
          }

      }

      return true;
    };
    /**
     * @internal
     */


    this.handleMenuItemClick = e => {
      if (e.defaultPrevented || this.disabled) {
        return false;
      }

      this.invoke();
      return false;
    };
    /**
     * @internal
     */


    this.submenuLoaded = () => {
      if (!this.focusSubmenuOnLoad) {
        return;
      }

      this.focusSubmenuOnLoad = false;

      if (this.hasSubmenu) {
        this.submenu.focus();
        this.setAttribute("tabindex", "-1");
      }
    };
    /**
     * @internal
     */


    this.handleMouseOver = e => {
      if (this.disabled || !this.hasSubmenu || this.expanded) {
        return false;
      }

      this.expanded = true;
      return false;
    };
    /**
     * @internal
     */


    this.handleMouseOut = e => {
      if (!this.expanded || this.contains(document.activeElement)) {
        return false;
      }

      this.expanded = false;
      return false;
    };
    /**
     * @internal
     */


    this.expandAndFocus = () => {
      if (!this.hasSubmenu) {
        return;
      }

      this.focusSubmenuOnLoad = true;
      this.expanded = true;
    };
    /**
     * @internal
     */


    this.invoke = () => {
      if (this.disabled) {
        return;
      }

      switch (this.role) {
        case MenuItemRole.menuitemcheckbox:
          this.checked = !this.checked;
          break;

        case MenuItemRole.menuitem:
          // update submenu
          this.updateSubmenu();

          if (this.hasSubmenu) {
            this.expandAndFocus();
          } else {
            this.$emit("change");
          }

          break;

        case MenuItemRole.menuitemradio:
          if (!this.checked) {
            this.checked = true;
          }

          break;
      }
    };
    /**
     * Gets the submenu element if any
     *
     * @internal
     */


    this.updateSubmenu = () => {
      this.submenu = this.domChildren().find(element => {
        return element.getAttribute("role") === "menu";
      });
      this.hasSubmenu = this.submenu === undefined ? false : true;
    };
  }

  expandedChanged(oldValue) {
    if (this.$fastController.isConnected) {
      if (this.submenu === undefined) {
        return;
      }

      if (this.expanded === false) {
        this.submenu.collapseExpandedItem();
      } else {
        this.currentDirection = getDirection(this);
      }

      this.$emit("expanded-change", this, {
        bubbles: false
      });
    }
  }

  checkedChanged(oldValue, newValue) {
    if (this.$fastController.isConnected) {
      this.$emit("change");
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    DOM.queueUpdate(() => {
      this.updateSubmenu();
    });

    if (!this.startColumnCount) {
      this.startColumnCount = 1;
    }

    this.observer = new MutationObserver(this.updateSubmenu);
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    super.disconnectedCallback();
    this.submenu = undefined;

    if (this.observer !== undefined) {
      this.observer.disconnect();
      this.observer = undefined;
    }
  }
  /**
   * get an array of valid DOM children
   */


  domChildren() {
    return Array.from(this.children).filter(child => !child.hasAttribute("hidden"));
  }

}

__decorate$1([attr({
  mode: "boolean"
})], MenuItem.prototype, "disabled", void 0);

__decorate$1([attr({
  mode: "boolean"
})], MenuItem.prototype, "expanded", void 0);

__decorate$1([observable], MenuItem.prototype, "startColumnCount", void 0);

__decorate$1([attr], MenuItem.prototype, "role", void 0);

__decorate$1([attr({
  mode: "boolean"
})], MenuItem.prototype, "checked", void 0);

__decorate$1([observable], MenuItem.prototype, "submenuRegion", void 0);

__decorate$1([observable], MenuItem.prototype, "hasSubmenu", void 0);

__decorate$1([observable], MenuItem.prototype, "currentDirection", void 0);

__decorate$1([observable], MenuItem.prototype, "submenu", void 0);

applyMixins(MenuItem, StartEnd);

/**
 * Generates a template for the {@link @microsoft/fast-foundation#(MenuItem:class)} component using
 * the provided prefix.
 *
 * @public
 */

const menuItemTemplate = (context, definition) => html`<template role="${x => x.role}" aria-haspopup="${x => x.hasSubmenu ? "menu" : void 0}" aria-checked="${x => x.role !== MenuItemRole.menuitem ? x.checked : void 0}" aria-disabled="${x => x.disabled}" aria-expanded="${x => x.expanded}" @keydown="${(x, c) => x.handleMenuItemKeyDown(c.event)}" @click="${(x, c) => x.handleMenuItemClick(c.event)}" @mouseover="${(x, c) => x.handleMouseOver(c.event)}" @mouseout="${(x, c) => x.handleMouseOut(c.event)}" class="${x => x.disabled ? "disabled" : ""} ${x => x.expanded ? "expanded" : ""} ${x => `indent-${x.startColumnCount}`}">${when(x => x.role === MenuItemRole.menuitemcheckbox, html`<div part="input-container" class="input-container"><span part="checkbox" class="checkbox"><slot name="checkbox-indicator">${definition.checkboxIndicator || ""}</slot></span></div>`)} ${when(x => x.role === MenuItemRole.menuitemradio, html`<div part="input-container" class="input-container"><span part="radio" class="radio"><slot name="radio-indicator">${definition.radioIndicator || ""}</slot></span></div>`)}</div>${startSlotTemplate(context, definition)}<span class="content" part="content"><slot></slot></span>${endSlotTemplate(context, definition)} ${when(x => x.hasSubmenu, html`<div part="expand-collapse-glyph-container" class="expand-collapse-glyph-container"><span part="expand-collapse" class="expand-collapse"><slot name="expand-collapse-indicator">${definition.expandCollapseGlyph || ""}</slot></span></div>`)} ${when(x => x.expanded, html`<${context.tagFor(AnchoredRegion)} :anchorElement="${x => x}" vertical-positioning-mode="dynamic" vertical-default-position="bottom" vertical-inset="true" horizontal-positioning-mode="dynamic" horizontal-default-position="end" class="submenu-region" dir="${x => x.currentDirection}" @loaded="${x => x.submenuLoaded()}" ${ref("submenuRegion")} part="submenu-region"><slot name="submenu"></slot></${context.tagFor(AnchoredRegion)}>`)}</template>`;

/**
 * The template for the {@link @microsoft/fast-foundation#Menu} component.
 * @public
 */

const menuTemplate = (context, definition) => html`<template slot="${x => x.slot ? x.slot : x.isNestedMenu() ? "submenu" : void 0}" role="menu" @keydown="${(x, c) => x.handleMenuKeyDown(c.event)}" @focusout="${(x, c) => x.handleFocusOut(c.event)}"><slot ${slotted("items")}></slot></template>`;

/**
 * A Menu Custom HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#menu | ARIA menu }.
 *
 * @slot - The default slot for the menu items
 *
 * @public
 */

class Menu$1 extends FoundationElement {
  constructor() {
    super(...arguments);
    this.expandedItem = null;
    /**
     * The index of the focusable element in the items array
     * defaults to -1
     */

    this.focusIndex = -1;
    /**
     * @internal
     */

    this.isNestedMenu = () => {
      return this.parentElement !== null && isHTMLElement(this.parentElement) && this.parentElement.getAttribute("role") === "menuitem";
    };
    /**
     * if focus is moving out of the menu, reset to a stable initial state
     * @internal
     */


    this.handleFocusOut = e => {
      if (!this.contains(e.relatedTarget) && this.menuItems !== undefined) {
        this.collapseExpandedItem(); // find our first focusable element

        const focusIndex = this.menuItems.findIndex(this.isFocusableElement); // set the current focus index's tabindex to -1

        this.menuItems[this.focusIndex].setAttribute("tabindex", "-1"); // set the first focusable element tabindex to 0

        this.menuItems[focusIndex].setAttribute("tabindex", "0"); // set the focus index

        this.focusIndex = focusIndex;
      }
    };

    this.handleItemFocus = e => {
      const targetItem = e.target;

      if (this.menuItems !== undefined && targetItem !== this.menuItems[this.focusIndex]) {
        this.menuItems[this.focusIndex].setAttribute("tabindex", "-1");
        this.focusIndex = this.menuItems.indexOf(targetItem);
        targetItem.setAttribute("tabindex", "0");
      }
    };

    this.handleExpandedChanged = e => {
      if (e.defaultPrevented || e.target === null || this.menuItems === undefined || this.menuItems.indexOf(e.target) < 0) {
        return;
      }

      e.preventDefault();
      const changedItem = e.target; // closing an expanded item without opening another

      if (this.expandedItem !== null && changedItem === this.expandedItem && changedItem.expanded === false) {
        this.expandedItem = null;
        return;
      }

      if (changedItem.expanded) {
        if (this.expandedItem !== null && this.expandedItem !== changedItem) {
          this.expandedItem.expanded = false;
        }

        this.menuItems[this.focusIndex].setAttribute("tabindex", "-1");
        this.expandedItem = changedItem;
        this.focusIndex = this.menuItems.indexOf(changedItem);
        changedItem.setAttribute("tabindex", "0");
      }
    };

    this.removeItemListeners = () => {
      if (this.menuItems !== undefined) {
        this.menuItems.forEach(item => {
          item.removeEventListener("expanded-change", this.handleExpandedChanged);
          item.removeEventListener("focus", this.handleItemFocus);
        });
      }
    };

    this.setItems = () => {
      const newItems = this.domChildren();
      this.removeItemListeners();
      this.menuItems = newItems;
      const menuItems = this.menuItems.filter(this.isMenuItemElement); // if our focus index is not -1 we have items

      if (menuItems.length) {
        this.focusIndex = 0;
      }

      function elementIndent(el) {
        const role = el.getAttribute("role");
        const startSlot = el.querySelector("[slot=start]");

        if (role !== MenuItemRole.menuitem && startSlot === null) {
          return 1;
        } else if (role === MenuItemRole.menuitem && startSlot !== null) {
          return 1;
        } else if (role !== MenuItemRole.menuitem && startSlot !== null) {
          return 2;
        } else {
          return 0;
        }
      }

      const indent = menuItems.reduce((accum, current) => {
        const elementValue = elementIndent(current);
        return accum > elementValue ? accum : elementValue;
      }, 0);
      menuItems.forEach((item, index) => {
        item.setAttribute("tabindex", index === 0 ? "0" : "-1");
        item.addEventListener("expanded-change", this.handleExpandedChanged);
        item.addEventListener("focus", this.handleItemFocus);

        if (item instanceof MenuItem) {
          item.startColumnCount = indent;
        }
      });
    };
    /**
     * handle change from child element
     */


    this.changeHandler = e => {
      if (this.menuItems === undefined) {
        return;
      }

      const changedMenuItem = e.target;
      const changeItemIndex = this.menuItems.indexOf(changedMenuItem);

      if (changeItemIndex === -1) {
        return;
      }

      if (changedMenuItem.role === "menuitemradio" && changedMenuItem.checked === true) {
        for (let i = changeItemIndex - 1; i >= 0; --i) {
          const item = this.menuItems[i];
          const role = item.getAttribute("role");

          if (role === MenuItemRole.menuitemradio) {
            item.checked = false;
          }

          if (role === "separator") {
            break;
          }
        }

        const maxIndex = this.menuItems.length - 1;

        for (let i = changeItemIndex + 1; i <= maxIndex; ++i) {
          const item = this.menuItems[i];
          const role = item.getAttribute("role");

          if (role === MenuItemRole.menuitemradio) {
            item.checked = false;
          }

          if (role === "separator") {
            break;
          }
        }
      }
    };
    /**
     * check if the item is a menu item
     */


    this.isMenuItemElement = el => {
      return isHTMLElement(el) && Menu$1.focusableElementRoles.hasOwnProperty(el.getAttribute("role"));
    };
    /**
     * check if the item is focusable
     */


    this.isFocusableElement = el => {
      return this.isMenuItemElement(el);
    };
  }

  itemsChanged(oldValue, newValue) {
    // only update children after the component is connected and
    // the setItems has run on connectedCallback
    // (menuItems is undefined until then)
    if (this.$fastController.isConnected && this.menuItems !== undefined) {
      this.setItems();
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    DOM.queueUpdate(() => {
      // wait until children have had a chance to
      // connect before setting/checking their props/attributes
      this.setItems();
    });
    this.addEventListener("change", this.changeHandler);
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    super.disconnectedCallback();
    this.removeItemListeners();
    this.menuItems = undefined;
    this.removeEventListener("change", this.changeHandler);
  }
  /**
   * Focuses the first item in the menu.
   *
   * @public
   */


  focus() {
    this.setFocus(0, 1);
  }
  /**
   * Collapses any expanded menu items.
   *
   * @public
   */


  collapseExpandedItem() {
    if (this.expandedItem !== null) {
      this.expandedItem.expanded = false;
      this.expandedItem = null;
    }
  }
  /**
   * @internal
   */


  handleMenuKeyDown(e) {
    if (e.defaultPrevented || this.menuItems === undefined) {
      return;
    }

    switch (e.key) {
      case keyArrowDown:
        // go forward one index
        this.setFocus(this.focusIndex + 1, 1);
        return;

      case keyArrowUp:
        // go back one index
        this.setFocus(this.focusIndex - 1, -1);
        return;

      case keyEnd:
        // set focus on last item
        this.setFocus(this.menuItems.length - 1, -1);
        return;

      case keyHome:
        // set focus on first item
        this.setFocus(0, 1);
        return;

      default:
        // if we are not handling the event, do not prevent default
        return true;
    }
  }
  /**
   * get an array of valid DOM children
   */


  domChildren() {
    return Array.from(this.children).filter(child => !child.hasAttribute("hidden"));
  }

  setFocus(focusIndex, adjustment) {
    if (this.menuItems === undefined) {
      return;
    }

    while (focusIndex >= 0 && focusIndex < this.menuItems.length) {
      const child = this.menuItems[focusIndex];

      if (this.isFocusableElement(child)) {
        // change the previous index to -1
        if (this.focusIndex > -1 && this.menuItems.length >= this.focusIndex - 1) {
          this.menuItems[this.focusIndex].setAttribute("tabindex", "-1");
        } // update the focus index


        this.focusIndex = focusIndex; // update the tabindex of next focusable element

        child.setAttribute("tabindex", "0"); // focus the element

        child.focus();
        break;
      }

      focusIndex += adjustment;
    }
  }

}
Menu$1.focusableElementRoles = roleForMenuItem;

__decorate$1([observable], Menu$1.prototype, "items", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(NumberField:class)} component.
 * @public
 */

const numberFieldTemplate = (context, definition) => html`<template class="${x => x.readOnly ? "readonly" : ""}"><label part="label" for="control" class="${x => x.defaultSlottedNodes && x.defaultSlottedNodes.length ? "label" : "label label__hidden"}"><slot ${slotted("defaultSlottedNodes")}></slot></label><div class="root" part="root">${startSlotTemplate(context, definition)}<input class="control" part="control" id="control" @input="${x => x.handleTextInput()}" @change="${x => x.handleChange()}" @keydown="${(x, c) => x.handleKeyDown(c.event)}" @blur="${(x, c) => x.handleBlur()}" ?autofocus="${x => x.autofocus}" ?disabled="${x => x.disabled}" list="${x => x.list}" maxlength="${x => x.maxlength}" minlength="${x => x.minlength}" placeholder="${x => x.placeholder}" ?readonly="${x => x.readOnly}" ?required="${x => x.required}" size="${x => x.size}" type="text" inputmode="numeric" min="${x => x.min}" max="${x => x.max}" step="${x => x.step}" aria-atomic="${x => x.ariaAtomic}" aria-busy="${x => x.ariaBusy}" aria-controls="${x => x.ariaControls}" aria-current="${x => x.ariaCurrent}" aria-describedby="${x => x.ariaDescribedby}" aria-details="${x => x.ariaDetails}" aria-disabled="${x => x.ariaDisabled}" aria-errormessage="${x => x.ariaErrormessage}" aria-flowto="${x => x.ariaFlowto}" aria-haspopup="${x => x.ariaHaspopup}" aria-hidden="${x => x.ariaHidden}" aria-invalid="${x => x.ariaInvalid}" aria-keyshortcuts="${x => x.ariaKeyshortcuts}" aria-label="${x => x.ariaLabel}" aria-labelledby="${x => x.ariaLabelledby}" aria-live="${x => x.ariaLive}" aria-owns="${x => x.ariaOwns}" aria-relevant="${x => x.ariaRelevant}" aria-roledescription="${x => x.ariaRoledescription}" ${ref("control")} />${when(x => !x.hideStep && !x.readOnly && !x.disabled, html`<div class="controls" part="controls"><div class="step-up" part="step-up" @click="${x => x.stepUp()}"><slot name="step-up-glyph">${definition.stepUpGlyph || ""}</slot></div><div class="step-down" part="step-down" @click="${x => x.stepDown()}"><slot name="step-down-glyph">${definition.stepDownGlyph || ""}</slot></div></div>`)} ${endSlotTemplate(context, definition)}</div></template>`;

class _TextField extends FoundationElement {}
/**
 * A form-associated base class for the {@link @microsoft/fast-foundation#(TextField:class)} component.
 *
 * @internal
 */


class FormAssociatedTextField extends FormAssociated(_TextField) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("input");
  }

}

/**
 * Text field sub-types
 * @public
 */
const TextFieldType = {
  /**
   * An email TextField
   */
  email: "email",

  /**
   * A password TextField
   */
  password: "password",

  /**
   * A telephone TextField
   */
  tel: "tel",

  /**
   * A text TextField
   */
  text: "text",

  /**
   * A URL TextField
   */
  url: "url"
};

/**
 * A Text Field Custom HTML Element.
 * Based largely on the {@link https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/text | <input type="text" /> element }.
 *
 * @slot start - Content which can be provided before the number field input
 * @slot end - Content which can be provided after the number field input
 * @slot - The default slot for the label
 * @csspart label - The label
 * @csspart root - The element wrapping the control, including start and end slots
 * @csspart control - The text field element
 * @fires change - Fires a custom 'change' event when the value has changed
 *
 * @public
 */

class TextField$1 extends FormAssociatedTextField {
  constructor() {
    super(...arguments);
    /**
     * Allows setting a type or mode of text.
     * @public
     * @remarks
     * HTML Attribute: type
     */

    this.type = TextFieldType.text;
  }

  readOnlyChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.readOnly = this.readOnly;
      this.validate();
    }
  }

  autofocusChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.autofocus = this.autofocus;
      this.validate();
    }
  }

  placeholderChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.placeholder = this.placeholder;
    }
  }

  typeChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.type = this.type;
      this.validate();
    }
  }

  listChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.setAttribute("list", this.list);
      this.validate();
    }
  }

  maxlengthChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.maxLength = this.maxlength;
      this.validate();
    }
  }

  minlengthChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.minLength = this.minlength;
      this.validate();
    }
  }

  patternChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.pattern = this.pattern;
      this.validate();
    }
  }

  sizeChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.size = this.size;
    }
  }

  spellcheckChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.spellcheck = this.spellcheck;
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.proxy.setAttribute("type", this.type);
    this.validate();

    if (this.autofocus) {
      DOM.queueUpdate(() => {
        this.focus();
      });
    }
  }
  /**
   * Selects all the text in the text field
   *
   * @public
   */


  select() {
    this.control.select();
    /**
     * The select event does not permeate the shadow DOM boundary.
     * This fn effectively proxies the select event,
     * emitting a `select` event whenever the internal
     * control emits a `select` event
     */

    this.$emit("select");
  }
  /**
   * Handles the internal control's `input` event
   * @internal
   */


  handleTextInput() {
    this.value = this.control.value;
  }
  /**
   * Change event handler for inner control.
   * @remarks
   * "Change" events are not `composable` so they will not
   * permeate the shadow DOM boundary. This fn effectively proxies
   * the change event, emitting a `change` event whenever the internal
   * control emits a `change` event
   * @internal
   */


  handleChange() {
    this.$emit("change");
  }

}

__decorate$1([attr({
  attribute: "readonly",
  mode: "boolean"
})], TextField$1.prototype, "readOnly", void 0);

__decorate$1([attr({
  mode: "boolean"
})], TextField$1.prototype, "autofocus", void 0);

__decorate$1([attr], TextField$1.prototype, "placeholder", void 0);

__decorate$1([attr], TextField$1.prototype, "type", void 0);

__decorate$1([attr], TextField$1.prototype, "list", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], TextField$1.prototype, "maxlength", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], TextField$1.prototype, "minlength", void 0);

__decorate$1([attr], TextField$1.prototype, "pattern", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], TextField$1.prototype, "size", void 0);

__decorate$1([attr({
  mode: "boolean"
})], TextField$1.prototype, "spellcheck", void 0);

__decorate$1([observable], TextField$1.prototype, "defaultSlottedNodes", void 0);
/**
 * Includes ARIA states and properties relating to the ARIA textbox role
 *
 * @public
 */


class DelegatesARIATextbox {}
applyMixins(DelegatesARIATextbox, ARIAGlobalStatesAndProperties);
applyMixins(TextField$1, StartEnd, DelegatesARIATextbox);

class _NumberField extends FoundationElement {}
/**
 * A form-associated base class for the {@link @microsoft/fast-foundation#(NumberField:class)} component.
 *
 * @internal
 */


class FormAssociatedNumberField extends FormAssociated(_NumberField) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("input");
  }

}

/**
 * A Number Field Custom HTML Element.
 * Based largely on the {@link https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/number | <input type="number" /> element }.
 *
 * @slot start - Content which can be provided before the number field input
 * @slot end - Content which can be provided after the number field input
 * @slot - The default slot for the label
 * @slot step-up-glyph - The glyph for the step up control
 * @slot step-down-glyph - The glyph for the step down control
 * @csspart label - The label
 * @csspart root - The element wrapping the control, including start and end slots
 * @csspart control - The element representing the input
 * @csspart controls - The step up and step down controls
 * @csspart step-up - The step up control
 * @csspart step-down - The step down control
 * @fires input - Fires a custom 'input' event when the value has changed
 * @fires change - Fires a custom 'change' event when the value has changed
 *
 * @public
 */

class NumberField$1 extends FormAssociatedNumberField {
  constructor() {
    super(...arguments);
    /**
     * When true, spin buttons will not be rendered
     * @public
     * @remarks
     * HTML Attribute: autofocus
     */

    this.hideStep = false;
    /**
     * Amount to increment or decrement the value by
     * @public
     * @remarks
     * HTMLAttribute: step
     */

    this.step = 1;
    /**
     * Flag to indicate that the value change is from the user input
     * @internal
     */

    this.isUserInput = false;
  }
  /**
   * Ensures that the max is greater than the min and that the value
   *  is less than the max
   * @param previous - the previous max value
   * @param next - updated max value
   *
   * @internal
   */


  maxChanged(previous, next) {
    var _a;

    this.max = Math.max(next, (_a = this.min) !== null && _a !== void 0 ? _a : next);
    const min = Math.min(this.min, this.max);

    if (this.min !== undefined && this.min !== min) {
      this.min = min;
    }

    this.value = this.getValidValue(this.value);
  }
  /**
   * Ensures that the min is less than the max and that the value
   *  is greater than the min
   * @param previous - previous min value
   * @param next - updated min value
   *
   * @internal
   */


  minChanged(previous, next) {
    var _a;

    this.min = Math.min(next, (_a = this.max) !== null && _a !== void 0 ? _a : next);
    const max = Math.max(this.min, this.max);

    if (this.max !== undefined && this.max !== max) {
      this.max = max;
    }

    this.value = this.getValidValue(this.value);
  }
  /**
   * The value property, typed as a number.
   *
   * @public
   */


  get valueAsNumber() {
    return parseFloat(super.value);
  }

  set valueAsNumber(next) {
    this.value = next.toString();
  }
  /**
   * Validates that the value is a number between the min and max
   * @param previous - previous stored value
   * @param next - value being updated
   * @param updateControl - should the text field be updated with value, defaults to true
   * @internal
   */


  valueChanged(previous, next) {
    this.value = this.getValidValue(next);

    if (next !== this.value) {
      return;
    }

    if (this.control && !this.isUserInput) {
      this.control.value = this.value;
    }

    super.valueChanged(previous, this.value);

    if (previous !== undefined && !this.isUserInput) {
      this.$emit("input");
      this.$emit("change");
    }

    this.isUserInput = false;
  }
  /**
   * Sets the internal value to a valid number between the min and max properties
   * @param value - user input
   *
   * @internal
   */


  getValidValue(value) {
    var _a, _b;

    let validValue = parseFloat(parseFloat(value).toPrecision(12));

    if (isNaN(validValue)) {
      validValue = "";
    } else {
      validValue = Math.min(validValue, (_a = this.max) !== null && _a !== void 0 ? _a : validValue);
      validValue = Math.max(validValue, (_b = this.min) !== null && _b !== void 0 ? _b : validValue).toString();
    }

    return validValue;
  }
  /**
   * Increments the value using the step value
   *
   * @public
   */


  stepUp() {
    const value = parseFloat(this.value);
    const stepUpValue = !isNaN(value) ? value + this.step : this.min > 0 ? this.min : this.max < 0 ? this.max : !this.min ? this.step : 0;
    this.value = stepUpValue.toString();
  }
  /**
   * Decrements the value using the step value
   *
   * @public
   */


  stepDown() {
    const value = parseFloat(this.value);
    const stepDownValue = !isNaN(value) ? value - this.step : this.min > 0 ? this.min : this.max < 0 ? this.max : !this.min ? 0 - this.step : 0;
    this.value = stepDownValue.toString();
  }
  /**
   * Sets up the initial state of the number field
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.proxy.setAttribute("type", "number");
    this.validate();
    this.control.value = this.value;

    if (this.autofocus) {
      DOM.queueUpdate(() => {
        this.focus();
      });
    }
  }
  /**
   * Selects all the text in the number field
   *
   * @public
   */


  select() {
    this.control.select();
    /**
     * The select event does not permeate the shadow DOM boundary.
     * This fn effectively proxies the select event,
     * emitting a `select` event whenever the internal
     * control emits a `select` event
     */

    this.$emit("select");
  }
  /**
   * Handles the internal control's `input` event
   * @internal
   */


  handleTextInput() {
    this.control.value = this.control.value.replace(/[^0-9\-+e.]/g, "");
    this.isUserInput = true;
    this.value = this.control.value;
  }
  /**
   * Change event handler for inner control.
   * @remarks
   * "Change" events are not `composable` so they will not
   * permeate the shadow DOM boundary. This fn effectively proxies
   * the change event, emitting a `change` event whenever the internal
   * control emits a `change` event
   * @internal
   */


  handleChange() {
    this.$emit("change");
  }
  /**
   * Handles the internal control's `keydown` event
   * @internal
   */


  handleKeyDown(e) {
    const key = e.key;

    switch (key) {
      case keyArrowUp:
        this.stepUp();
        return false;

      case keyArrowDown:
        this.stepDown();
        return false;
    }

    return true;
  }
  /**
   * Handles populating the input field with a validated value when
   *  leaving the input field.
   * @internal
   */


  handleBlur() {
    this.control.value = this.value;
  }

}

__decorate$1([attr({
  attribute: "readonly",
  mode: "boolean"
})], NumberField$1.prototype, "readOnly", void 0);

__decorate$1([attr({
  mode: "boolean"
})], NumberField$1.prototype, "autofocus", void 0);

__decorate$1([attr({
  attribute: "hide-step",
  mode: "boolean"
})], NumberField$1.prototype, "hideStep", void 0);

__decorate$1([attr], NumberField$1.prototype, "placeholder", void 0);

__decorate$1([attr], NumberField$1.prototype, "list", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], NumberField$1.prototype, "maxlength", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], NumberField$1.prototype, "minlength", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], NumberField$1.prototype, "size", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], NumberField$1.prototype, "step", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], NumberField$1.prototype, "max", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], NumberField$1.prototype, "min", void 0);

__decorate$1([observable], NumberField$1.prototype, "defaultSlottedNodes", void 0);

applyMixins(NumberField$1, StartEnd, DelegatesARIATextbox);

const progressSegments = 44;
/**
 * The template for the {@link @microsoft/fast-foundation#BaseProgress} component.
 * @public
 */

const progressRingTemplate = (context, definition) => html`<template role="progressbar" aria-valuenow="${x => x.value}" aria-valuemin="${x => x.min}" aria-valuemax="${x => x.max}" class="${x => x.paused ? "paused" : ""}">${when(x => typeof x.value === "number", html`<svg class="progress" part="progress" viewBox="0 0 16 16" slot="determinate"><circle class="background" part="background" cx="8px" cy="8px" r="7px"></circle><circle class="determinate" part="determinate" style="stroke-dasharray: ${x => progressSegments * x.percentComplete / 100}px ${progressSegments}px" cx="8px" cy="8px" r="7px"></circle></svg>`)} ${when(x => typeof x.value !== "number", html`<slot name="indeterminate" slot="indeterminate">${definition.indeterminateIndicator || ""}</slot>`)}</template>`;

/**
 * An Progress HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#progressbar | ARIA progressbar }.
 *
 * @slot indeterminate - The slot for a custom indeterminate indicator
 * @csspart progress - Represents the progress element
 * @csspart determinate - The determinate indicator
 * @csspart indeterminate - The indeterminate indicator
 *
 * @public
 */

class BaseProgress extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * Indicates progress in %
     * @internal
     */

    this.percentComplete = 0;
  }

  valueChanged() {
    if (this.$fastController.isConnected) {
      this.updatePercentComplete();
    }
  }

  minChanged() {
    if (this.$fastController.isConnected) {
      this.updatePercentComplete();
    }
  }

  maxChanged() {
    if (this.$fastController.isConnected) {
      this.updatePercentComplete();
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.updatePercentComplete();
  }

  updatePercentComplete() {
    const min = typeof this.min === "number" ? this.min : 0;
    const max = typeof this.max === "number" ? this.max : 100;
    const value = typeof this.value === "number" ? this.value : 0;
    const range = max - min;
    this.percentComplete = range === 0 ? 0 : Math.fround((value - min) / range * 100);
  }

}

__decorate$1([attr({
  converter: nullableNumberConverter
})], BaseProgress.prototype, "value", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], BaseProgress.prototype, "min", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], BaseProgress.prototype, "max", void 0);

__decorate$1([attr({
  mode: "boolean"
})], BaseProgress.prototype, "paused", void 0);

__decorate$1([observable], BaseProgress.prototype, "percentComplete", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#BaseProgress} component.
 * @public
 */

const progressTemplate = (context, defintion) => html`<template role="progressbar" aria-valuenow="${x => x.value}" aria-valuemin="${x => x.min}" aria-valuemax="${x => x.max}" class="${x => x.paused ? "paused" : ""}">${when(x => typeof x.value === "number", html`<div class="progress" part="progress" slot="determinate"><div class="determinate" part="determinate" style="width: ${x => x.percentComplete}%"></div></div>`)} ${when(x => typeof x.value !== "number", html`<div class="progress" part="progress" slot="indeterminate"><slot class="indeterminate" name="indeterminate">${defintion.indeterminateIndicator1 || ""} ${defintion.indeterminateIndicator2 || ""}</slot></div>`)}</template>`;

/**
 * The template for the {@link @microsoft/fast-foundation#RadioGroup} component.
 * @public
 */

const radioGroupTemplate = (context, definition) => html`<template role="radiogroup" aria-disabled="${x => x.disabled}" aria-readonly="${x => x.readOnly}" @click="${(x, c) => x.clickHandler(c.event)}" @keydown="${(x, c) => x.keydownHandler(c.event)}" @focusout="${(x, c) => x.focusOutHandler(c.event)}"><slot name="label"></slot><div class="positioning-region ${x => x.orientation === Orientation.horizontal ? "horizontal" : "vertical"}" part="positioning-region"><slot ${slotted({
  property: "slottedRadioButtons",
  filter: elements("[role=radio]")
})}></slot></div></template>`;

/**
 * An Radio Group Custom HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#radiogroup | ARIA radiogroup }.
 *
 * @slot label - The slot for the label
 * @slot - The default slot for radio buttons
 * @csspart positioning-region - The positioning region for laying out the radios
 * @fires change - Fires a custom 'change' event when the value changes
 *
 * @public
 */

class RadioGroup extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * The orientation of the group
     *
     * @public
     * @remarks
     * HTML Attribute: orientation
     */

    this.orientation = Orientation.horizontal;

    this.radioChangeHandler = e => {
      const changedRadio = e.target;

      if (changedRadio.checked) {
        this.slottedRadioButtons.forEach(radio => {
          if (radio !== changedRadio) {
            radio.checked = false;

            if (!this.isInsideFoundationToolbar) {
              radio.setAttribute("tabindex", "-1");
            }
          }
        });
        this.selectedRadio = changedRadio;
        this.value = changedRadio.value;
        changedRadio.setAttribute("tabindex", "0");
        this.focusedRadio = changedRadio;
      }

      e.stopPropagation();
    };

    this.moveToRadioByIndex = (group, index) => {
      const radio = group[index];

      if (!this.isInsideToolbar) {
        radio.setAttribute("tabindex", "0");

        if (radio.readOnly) {
          this.slottedRadioButtons.forEach(nextRadio => {
            if (nextRadio !== radio) {
              nextRadio.setAttribute("tabindex", "-1");
            }
          });
        } else {
          radio.checked = true;
          this.selectedRadio = radio;
        }
      }

      this.focusedRadio = radio;
      radio.focus();
    };

    this.moveRightOffGroup = () => {
      var _a;

      (_a = this.nextElementSibling) === null || _a === void 0 ? void 0 : _a.focus();
    };

    this.moveLeftOffGroup = () => {
      var _a;

      (_a = this.previousElementSibling) === null || _a === void 0 ? void 0 : _a.focus();
    };
    /**
     * @internal
     */


    this.focusOutHandler = e => {
      const group = this.slottedRadioButtons;
      const radio = e.target;
      const index = radio !== null ? group.indexOf(radio) : 0;
      const focusedIndex = this.focusedRadio ? group.indexOf(this.focusedRadio) : -1;

      if (focusedIndex === 0 && index === focusedIndex || focusedIndex === group.length - 1 && focusedIndex === index) {
        if (!this.selectedRadio) {
          this.focusedRadio = group[0];
          this.focusedRadio.setAttribute("tabindex", "0");
          group.forEach(nextRadio => {
            if (nextRadio !== this.focusedRadio) {
              nextRadio.setAttribute("tabindex", "-1");
            }
          });
        } else {
          this.focusedRadio = this.selectedRadio;

          if (!this.isInsideFoundationToolbar) {
            this.selectedRadio.setAttribute("tabindex", "0");
            group.forEach(nextRadio => {
              if (nextRadio !== this.selectedRadio) {
                nextRadio.setAttribute("tabindex", "-1");
              }
            });
          }
        }
      }

      return true;
    };
    /**
     * @internal
     */


    this.clickHandler = e => {
      const radio = e.target;

      if (radio) {
        const group = this.slottedRadioButtons;

        if (radio.checked || group.indexOf(radio) === 0) {
          radio.setAttribute("tabindex", "0");
          this.selectedRadio = radio;
        } else {
          radio.setAttribute("tabindex", "-1");
          this.selectedRadio = null;
        }

        this.focusedRadio = radio;
      }

      e.preventDefault();
    };

    this.shouldMoveOffGroupToTheRight = (index, group, key) => {
      return index === group.length && this.isInsideToolbar && key === keyArrowRight;
    };

    this.shouldMoveOffGroupToTheLeft = (group, key) => {
      const index = this.focusedRadio ? group.indexOf(this.focusedRadio) - 1 : 0;
      return index < 0 && this.isInsideToolbar && key === keyArrowLeft;
    };

    this.checkFocusedRadio = () => {
      if (this.focusedRadio !== null && !this.focusedRadio.readOnly && !this.focusedRadio.checked) {
        this.focusedRadio.checked = true;
        this.focusedRadio.setAttribute("tabindex", "0");
        this.focusedRadio.focus();
        this.selectedRadio = this.focusedRadio;
      }
    };

    this.moveRight = e => {
      const group = this.slottedRadioButtons;
      let index = 0;
      index = this.focusedRadio ? group.indexOf(this.focusedRadio) + 1 : 1;

      if (this.shouldMoveOffGroupToTheRight(index, group, e.key)) {
        this.moveRightOffGroup();
        return;
      } else if (index === group.length) {
        index = 0;
      }
      /* looping to get to next radio that is not disabled */

      /* matching native radio/radiogroup which does not select an item if there is only 1 in the group */


      while (index < group.length && group.length > 1) {
        if (!group[index].disabled) {
          this.moveToRadioByIndex(group, index);
          break;
        } else if (this.focusedRadio && index === group.indexOf(this.focusedRadio)) {
          break;
        } else if (index + 1 >= group.length) {
          if (this.isInsideToolbar) {
            break;
          } else {
            index = 0;
          }
        } else {
          index += 1;
        }
      }
    };

    this.moveLeft = e => {
      const group = this.slottedRadioButtons;
      let index = 0;
      index = this.focusedRadio ? group.indexOf(this.focusedRadio) - 1 : 0;
      index = index < 0 ? group.length - 1 : index;

      if (this.shouldMoveOffGroupToTheLeft(group, e.key)) {
        this.moveLeftOffGroup();
        return;
      }
      /* looping to get to next radio that is not disabled */


      while (index >= 0 && group.length > 1) {
        if (!group[index].disabled) {
          this.moveToRadioByIndex(group, index);
          break;
        } else if (this.focusedRadio && index === group.indexOf(this.focusedRadio)) {
          break;
        } else if (index - 1 < 0) {
          index = group.length - 1;
        } else {
          index -= 1;
        }
      }
    };
    /**
     * keyboard handling per https://w3c.github.io/aria-practices/#for-radio-groups-not-contained-in-a-toolbar
     * navigation is different when there is an ancestor with role='toolbar'
     *
     * @internal
     */


    this.keydownHandler = e => {
      const key = e.key;

      if (key in ArrowKeys && this.isInsideFoundationToolbar) {
        return true;
      }

      switch (key) {
        case keyEnter:
          {
            this.checkFocusedRadio();
            break;
          }

        case keyArrowRight:
        case keyArrowDown:
          {
            if (this.direction === Direction.ltr) {
              this.moveRight(e);
            } else {
              this.moveLeft(e);
            }

            break;
          }

        case keyArrowLeft:
        case keyArrowUp:
          {
            if (this.direction === Direction.ltr) {
              this.moveLeft(e);
            } else {
              this.moveRight(e);
            }

            break;
          }

        default:
          {
            return true;
          }
      }
    };
  }

  readOnlyChanged() {
    if (this.slottedRadioButtons !== undefined) {
      this.slottedRadioButtons.forEach(radio => {
        if (this.readOnly) {
          radio.readOnly = true;
        } else {
          radio.readOnly = false;
        }
      });
    }
  }

  disabledChanged() {
    if (this.slottedRadioButtons !== undefined) {
      this.slottedRadioButtons.forEach(radio => {
        if (this.disabled) {
          radio.disabled = true;
        } else {
          radio.disabled = false;
        }
      });
    }
  }

  nameChanged() {
    if (this.slottedRadioButtons) {
      this.slottedRadioButtons.forEach(radio => {
        radio.setAttribute("name", this.name);
      });
    }
  }

  valueChanged() {
    if (this.slottedRadioButtons) {
      this.slottedRadioButtons.forEach(radio => {
        if (radio.getAttribute("value") === this.value) {
          radio.checked = true;
          this.selectedRadio = radio;
        }
      });
    }

    this.$emit("change");
  }

  slottedRadioButtonsChanged(oldValue, newValue) {
    if (this.slottedRadioButtons && this.slottedRadioButtons.length > 0) {
      this.setupRadioButtons();
    }
  }

  get parentToolbar() {
    return this.closest('[role="toolbar"]');
  }

  get isInsideToolbar() {
    var _a;

    return (_a = this.parentToolbar) !== null && _a !== void 0 ? _a : false;
  }

  get isInsideFoundationToolbar() {
    var _a;

    return !!((_a = this.parentToolbar) === null || _a === void 0 ? void 0 : _a["$fastController"]);
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.direction = getDirection(this);
    this.setupRadioButtons();
  }

  disconnectedCallback() {
    this.slottedRadioButtons.forEach(radio => {
      radio.removeEventListener("change", this.radioChangeHandler);
    });
  }

  setupRadioButtons() {
    const checkedRadios = this.slottedRadioButtons.filter(radio => {
      return radio.hasAttribute("checked");
    });
    const numberOfCheckedRadios = checkedRadios ? checkedRadios.length : 0;

    if (numberOfCheckedRadios > 1) {
      const lastCheckedRadio = checkedRadios[numberOfCheckedRadios - 1];
      lastCheckedRadio.checked = true;
    }

    let foundMatchingVal = false;
    this.slottedRadioButtons.forEach(radio => {
      if (this.name !== undefined) {
        radio.setAttribute("name", this.name);
      }

      if (this.disabled) {
        radio.disabled = true;
      }

      if (this.readOnly) {
        radio.readOnly = true;
      }

      if (this.value && this.value === radio.value) {
        this.selectedRadio = radio;
        this.focusedRadio = radio;
        radio.checked = true;
        radio.setAttribute("tabindex", "0");
        foundMatchingVal = true;
      } else {
        if (!this.isInsideFoundationToolbar) {
          radio.setAttribute("tabindex", "-1");
        }

        radio.checked = false;
      }

      radio.addEventListener("change", this.radioChangeHandler);
    });

    if (this.value === undefined && this.slottedRadioButtons.length > 0) {
      const checkedRadios = this.slottedRadioButtons.filter(radio => {
        return radio.hasAttribute("checked");
      });
      const numberOfCheckedRadios = checkedRadios !== null ? checkedRadios.length : 0;

      if (numberOfCheckedRadios > 0 && !foundMatchingVal) {
        const lastCheckedRadio = checkedRadios[numberOfCheckedRadios - 1];
        lastCheckedRadio.checked = true;
        this.focusedRadio = lastCheckedRadio;
        lastCheckedRadio.setAttribute("tabindex", "0");
      } else {
        this.slottedRadioButtons[0].setAttribute("tabindex", "0");
        this.focusedRadio = this.slottedRadioButtons[0];
      }
    }
  }

}

__decorate$1([attr({
  attribute: "readonly",
  mode: "boolean"
})], RadioGroup.prototype, "readOnly", void 0);

__decorate$1([attr({
  attribute: "disabled",
  mode: "boolean"
})], RadioGroup.prototype, "disabled", void 0);

__decorate$1([attr], RadioGroup.prototype, "name", void 0);

__decorate$1([attr], RadioGroup.prototype, "value", void 0);

__decorate$1([attr], RadioGroup.prototype, "orientation", void 0);

__decorate$1([observable], RadioGroup.prototype, "childItems", void 0);

__decorate$1([observable], RadioGroup.prototype, "slottedRadioButtons", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(Radio:class)} component.
 * @public
 */

const radioTemplate = (context, definition) => html`<template role="radio" class="${x => x.checked ? "checked" : ""} ${x => x.readOnly ? "readonly" : ""}" aria-checked="${x => x.checked}" aria-required="${x => x.required}" aria-disabled="${x => x.disabled}" aria-readonly="${x => x.readOnly}" @keypress="${(x, c) => x.keypressHandler(c.event)}" @click="${(x, c) => x.clickHandler(c.event)}"><div part="control" class="control"><slot name="checked-indicator">${definition.checkedIndicator || ""}</slot></div><label part="label" class="${x => x.defaultSlottedNodes && x.defaultSlottedNodes.length ? "label" : "label label__hidden"}"><slot ${slotted("defaultSlottedNodes")}></slot></label></template>`;

class _Radio extends FoundationElement {}
/**
 * A form-associated base class for the {@link @microsoft/fast-foundation#(Radio:class)} component.
 *
 * @internal
 */


class FormAssociatedRadio extends CheckableFormAssociated(_Radio) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("input");
  }

}

/**
 * A Radio Custom HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#radio | ARIA radio }.
 *
 * @slot checked-indicator - The checked indicator
 * @slot - The default slot for the label
 * @csspart control - The element representing the visual radio control
 * @csspart label - The label
 * @fires change - Emits a custom change event when the checked state changes
 *
 * @public
 */

class Radio extends FormAssociatedRadio {
  constructor() {
    super();
    /**
     * The element's value to be included in form submission when checked.
     * Default to "on" to reach parity with input[type="radio"]
     *
     * @internal
     */

    this.initialValue = "on";
    /**
     * @internal
     */

    this.keypressHandler = e => {
      switch (e.key) {
        case keySpace:
          if (!this.checked && !this.readOnly) {
            this.checked = true;
          }

          return;
      }

      return true;
    };

    this.proxy.setAttribute("type", "radio");
  }

  readOnlyChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.readOnly = this.readOnly;
    }
  }
  /**
   * @internal
   */


  defaultCheckedChanged() {
    var _a;

    if (this.$fastController.isConnected && !this.dirtyChecked) {
      // Setting this.checked will cause us to enter a dirty state,
      // but if we are clean when defaultChecked is changed, we want to stay
      // in a clean state, so reset this.dirtyChecked
      if (!this.isInsideRadioGroup()) {
        this.checked = (_a = this.defaultChecked) !== null && _a !== void 0 ? _a : false;
        this.dirtyChecked = false;
      }
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    var _a, _b;

    super.connectedCallback();
    this.validate();

    if (((_a = this.parentElement) === null || _a === void 0 ? void 0 : _a.getAttribute("role")) !== "radiogroup" && this.getAttribute("tabindex") === null) {
      if (!this.disabled) {
        this.setAttribute("tabindex", "0");
      }
    }

    if (this.checkedAttribute) {
      if (!this.dirtyChecked) {
        // Setting this.checked will cause us to enter a dirty state,
        // but if we are clean when defaultChecked is changed, we want to stay
        // in a clean state, so reset this.dirtyChecked
        if (!this.isInsideRadioGroup()) {
          this.checked = (_b = this.defaultChecked) !== null && _b !== void 0 ? _b : false;
          this.dirtyChecked = false;
        }
      }
    }
  }

  isInsideRadioGroup() {
    const parent = this.closest("[role=radiogroup]");
    return parent !== null;
  }
  /**
   * @internal
   */


  clickHandler(e) {
    if (!this.disabled && !this.readOnly && !this.checked) {
      this.checked = true;
    }
  }

}

__decorate$1([attr({
  attribute: "readonly",
  mode: "boolean"
})], Radio.prototype, "readOnly", void 0);

__decorate$1([observable], Radio.prototype, "name", void 0);

__decorate$1([observable], Radio.prototype, "defaultSlottedNodes", void 0);

/**
 * A HorizontalScroll Custom HTML Element
 *
 * @slot start - Content which can be provided before the scroll area
 * @slot end - Content which can be provided after the scroll area
 * @csspart scroll-area - Wraps the entire scrollable region
 * @csspart scroll-view - The visible scroll area
 * @csspart content-container - The container for the content
 * @csspart scroll-prev - The previous flipper container
 * @csspart scroll-action-previous - The element wrapping the previous flipper
 * @csspart scroll-next - The next flipper container
 * @csspart scroll-action-next - The element wrapping the next flipper
 * @fires scrollstart - Fires a custom 'scrollstart' event when scrolling
 * @fires scrollend - Fires a custom 'scrollend' event when scrolling stops
 *
 * @public
 */

class HorizontalScroll$1 extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * @internal
     */

    this.framesPerSecond = 60;
    /**
     * Flag indicating that the items are being updated
     *
     * @internal
     */

    this.updatingItems = false;
    /**
     * Speed of scroll in pixels per second
     * @public
     */

    this.speed = 600;
    /**
     * Attribute used for easing, defaults to ease-in-out
     * @public
     */

    this.easing = "ease-in-out";
    /**
     * Attribute to hide flippers from assistive technology
     * @public
     */

    this.flippersHiddenFromAT = false;
    /**
     * Scrolling state
     * @internal
     */

    this.scrolling = false;
    /**
     * Detects if the component has been resized
     * @internal
     */

    this.resizeDetector = null;
  }
  /**
   * The calculated duration for a frame.
   *
   * @internal
   */


  get frameTime() {
    return 1000 / this.framesPerSecond;
  }
  /**
   * Firing scrollstart and scrollend events
   * @internal
   */


  scrollingChanged(prev, next) {
    if (this.scrollContainer) {
      const event = this.scrolling == true ? "scrollstart" : "scrollend";
      this.$emit(event, this.scrollContainer.scrollLeft);
    }
  }
  /**
   * In RTL mode
   * @internal
   */


  get isRtl() {
    return this.scrollItems.length > 1 && this.scrollItems[0].offsetLeft > this.scrollItems[1].offsetLeft;
  }

  connectedCallback() {
    super.connectedCallback();
    this.initializeResizeDetector();
  }

  disconnectedCallback() {
    this.disconnectResizeDetector();
    super.disconnectedCallback();
  }
  /**
   * Updates scroll stops and flippers when scroll items change
   * @param previous - current scroll items
   * @param next - new updated scroll items
   * @public
   */


  scrollItemsChanged(previous, next) {
    if (next && !this.updatingItems) {
      DOM.queueUpdate(() => this.setStops());
    }
  }
  /**
   * destroys the instance's resize observer
   * @internal
   */


  disconnectResizeDetector() {
    if (this.resizeDetector) {
      this.resizeDetector.disconnect();
      this.resizeDetector = null;
    }
  }
  /**
   * initializes the instance's resize observer
   * @internal
   */


  initializeResizeDetector() {
    this.disconnectResizeDetector();
    this.resizeDetector = new window.ResizeObserver(this.resized.bind(this));
    this.resizeDetector.observe(this);
  }
  /**
   * Looks for slots and uses child nodes instead
   * @internal
   */


  updateScrollStops() {
    this.updatingItems = true;
    const updatedItems = this.scrollItems.reduce((scrollItems, scrollItem) => {
      if (scrollItem instanceof HTMLSlotElement) {
        return scrollItems.concat(scrollItem.assignedElements());
      }

      scrollItems.push(scrollItem);
      return scrollItems;
    }, []);
    this.scrollItems = updatedItems;
    this.updatingItems = false;
  }
  /**
   * Finds all of the scroll stops between elements
   * @internal
   */


  setStops() {
    this.updateScrollStops();
    this.width = this.offsetWidth;
    let lastStop = 0;
    let stops = this.scrollItems.map(({
      offsetLeft: left,
      offsetWidth: width
    }, index) => {
      const right = left + width;

      if (this.isRtl) {
        return -right;
      }

      lastStop = right;
      return index === 0 ? 0 : left;
    }).concat(lastStop);
    /* Fixes a FireFox bug where it doesn't scroll to the start */

    stops = this.fixScrollMisalign(stops);
    /* Sort to zero */

    stops.sort((a, b) => Math.abs(a) - Math.abs(b));
    this.scrollStops = stops;
    this.setFlippers();
  }
  /**
   *
   */


  fixScrollMisalign(stops) {
    if (this.isRtl && stops.some(stop => stop > 0)) {
      stops.sort((a, b) => b - a);
      const offset = stops[0];
      stops = stops.map(stop => stop - offset);
    }

    return stops;
  }
  /**
   * Sets the controls view if enabled
   * @internal
   */


  setFlippers() {
    var _a, _b;

    const position = this.scrollContainer.scrollLeft;
    (_a = this.previousFlipperContainer) === null || _a === void 0 ? void 0 : _a.classList.toggle("disabled", position === 0);

    if (this.scrollStops) {
      const lastStop = Math.abs(this.scrollStops[this.scrollStops.length - 1]);
      (_b = this.nextFlipperContainer) === null || _b === void 0 ? void 0 : _b.classList.toggle("disabled", Math.abs(position) + this.width >= lastStop);
    }
  }
  /**
   * Lets the user arrow left and right through the horizontal scroll
   * @param e - Keyboard event
   * @public
   */


  keyupHandler(e) {
    const key = e.key;

    switch (key) {
      case "ArrowLeft":
        this.scrollToPrevious();
        break;

      case "ArrowRight":
        this.scrollToNext();
        break;
    }
  }
  /**
   * Scrolls items to the left
   * @public
   */


  scrollToPrevious() {
    const scrollPosition = this.scrollContainer.scrollLeft;
    const current = this.scrollStops.findIndex((stop, index) => stop >= scrollPosition && (this.isRtl || index === this.scrollStops.length - 1 || this.scrollStops[index + 1] > scrollPosition));
    const right = Math.abs(this.scrollStops[current + 1]);
    let nextIndex = this.scrollStops.findIndex(stop => Math.abs(stop) + this.width > right);

    if (nextIndex >= current || nextIndex === -1) {
      nextIndex = current > 0 ? current - 1 : 0;
    }

    this.scrollToPosition(this.scrollStops[nextIndex], scrollPosition);
  }
  /**
   * Scrolls items to the right
   * @public
   */


  scrollToNext() {
    const scrollPosition = this.scrollContainer.scrollLeft;
    const current = this.scrollStops.findIndex(stop => Math.abs(stop) >= Math.abs(scrollPosition));
    const outOfView = this.scrollStops.findIndex(stop => Math.abs(scrollPosition) + this.width <= Math.abs(stop));
    let nextIndex = current;

    if (outOfView > current + 2) {
      nextIndex = outOfView - 2;
    } else if (current < this.scrollStops.length - 2) {
      nextIndex = current + 1;
    }

    this.scrollToPosition(this.scrollStops[nextIndex], scrollPosition);
  }
  /**
   * Handles scrolling with easing
   * @param position - starting position
   * @param newPosition - position to scroll to
   * @public
   */


  scrollToPosition(newPosition, position = this.scrollContainer.scrollLeft) {
    var _a;

    if (this.scrolling) {
      return;
    }

    this.scrolling = true;
    const seconds = (_a = this.duration) !== null && _a !== void 0 ? _a : `${Math.abs(newPosition - position) / this.speed}s`;
    this.content.style.setProperty("transition-duration", seconds);
    const computedDuration = parseFloat(getComputedStyle(this.content).getPropertyValue("transition-duration"));

    const transitionendHandler = e => {
      if (e && e.target !== e.currentTarget) {
        return;
      }

      this.content.style.setProperty("transition-duration", "0s");
      this.content.style.removeProperty("transform");
      this.scrollContainer.style.setProperty("scroll-behavior", "auto");
      this.scrollContainer.scrollLeft = newPosition;
      this.setFlippers();
      this.content.removeEventListener("transitionend", transitionendHandler);
      this.scrolling = false;
    };

    if (computedDuration === 0) {
      transitionendHandler();
      return;
    }

    this.content.addEventListener("transitionend", transitionendHandler);
    const maxScrollValue = this.scrollContainer.scrollWidth - this.scrollContainer.clientWidth;
    let transitionStop = this.scrollContainer.scrollLeft - Math.min(newPosition, maxScrollValue);

    if (this.isRtl) {
      transitionStop = this.scrollContainer.scrollLeft + Math.min(Math.abs(newPosition), maxScrollValue);
    }

    this.content.style.setProperty("transition-property", "transform");
    this.content.style.setProperty("transition-timing-function", this.easing);
    this.content.style.setProperty("transform", `translateX(${transitionStop}px)`);
  }
  /**
   * Monitors resize event on the horizontal-scroll element
   * @public
   */


  resized() {
    if (this.resizeTimeout) {
      this.resizeTimeout = clearTimeout(this.resizeTimeout);
    }

    this.resizeTimeout = setTimeout(() => {
      this.width = this.offsetWidth;
      this.setFlippers();
    }, this.frameTime);
  }
  /**
   * Monitors scrolled event on the content container
   * @public
   */


  scrolled() {
    if (this.scrollTimeout) {
      this.scrollTimeout = clearTimeout(this.scrollTimeout);
    }

    this.scrollTimeout = setTimeout(() => {
      this.setFlippers();
    }, this.frameTime);
  }

}

__decorate$1([attr({
  converter: nullableNumberConverter
})], HorizontalScroll$1.prototype, "speed", void 0);

__decorate$1([attr], HorizontalScroll$1.prototype, "duration", void 0);

__decorate$1([attr], HorizontalScroll$1.prototype, "easing", void 0);

__decorate$1([attr({
  attribute: "flippers-hidden-from-at",
  converter: booleanConverter
})], HorizontalScroll$1.prototype, "flippersHiddenFromAT", void 0);

__decorate$1([observable], HorizontalScroll$1.prototype, "scrolling", void 0);

__decorate$1([observable], HorizontalScroll$1.prototype, "scrollItems", void 0);

__decorate$1([attr({
  attribute: "view"
})], HorizontalScroll$1.prototype, "view", void 0);

/**
 * @public
 */

const horizontalScrollTemplate = (context, definition) => {
  var _a, _b;

  return html`<template class="horizontal-scroll" @keyup="${(x, c) => x.keyupHandler(c.event)}">${startSlotTemplate(context, definition)}<div class="scroll-area" part="scroll-area"><div class="scroll-view" part="scroll-view" @scroll="${x => x.scrolled()}" ${ref("scrollContainer")}><div class="content-container" part="content-container" ${ref("content")}><slot ${slotted({
    property: "scrollItems",
    filter: elements()
  })}></slot></div></div>${when(x => x.view !== "mobile", html`<div class="scroll scroll-prev" part="scroll-prev" ${ref("previousFlipperContainer")}><div class="scroll-action" part="scroll-action-previous"><slot name="previous-flipper">${definition.previousFlipper instanceof Function ? definition.previousFlipper(context, definition) : (_a = definition.previousFlipper) !== null && _a !== void 0 ? _a : ""}</slot></div></div><div class="scroll scroll-next" part="scroll-next" ${ref("nextFlipperContainer")}><div class="scroll-action" part="scroll-action-next"><slot name="next-flipper">${definition.nextFlipper instanceof Function ? definition.nextFlipper(context, definition) : (_b = definition.nextFlipper) !== null && _b !== void 0 ? _b : ""}</slot></div></div>`)}</div>${endSlotTemplate(context, definition)}</template>`;
};

/**
 * a method to filter out any whitespace _only_ nodes, to be used inside a template
 * @param value - The Node that is being inspected
 * @param index - The index of the node within the array
 * @param array - The Node array that is being filtered
 *
 * @public
 */
function whitespaceFilter(value, index, array) {
  return value.nodeType !== Node.TEXT_NODE ? true : typeof value.nodeValue === "string" && !!value.nodeValue.trim().length;
}

/**
 * The template for the {@link @microsoft/fast-foundation#(Search:class)} component.
 * @public
 */

const searchTemplate = (context, definition) => html`<template class=" ${x => x.readOnly ? "readonly" : ""} "><label part="label" for="control" class="${x => x.defaultSlottedNodes && x.defaultSlottedNodes.length ? "label" : "label label__hidden"}"><slot ${slotted({
  property: "defaultSlottedNodes",
  filter: whitespaceFilter
})}></slot></label><div class="root" part="root" ${ref("root")}>${startSlotTemplate(context, definition)}<div class="input-wrapper" part="input-wrapper"><input class="control" part="control" id="control" @input="${x => x.handleTextInput()}" @change="${x => x.handleChange()}" ?autofocus="${x => x.autofocus}" ?disabled="${x => x.disabled}" list="${x => x.list}" maxlength="${x => x.maxlength}" minlength="${x => x.minlength}" pattern="${x => x.pattern}" placeholder="${x => x.placeholder}" ?readonly="${x => x.readOnly}" ?required="${x => x.required}" size="${x => x.size}" ?spellcheck="${x => x.spellcheck}" :value="${x => x.value}" type="search" aria-atomic="${x => x.ariaAtomic}" aria-busy="${x => x.ariaBusy}" aria-controls="${x => x.ariaControls}" aria-current="${x => x.ariaCurrent}" aria-describedby="${x => x.ariaDescribedby}" aria-details="${x => x.ariaDetails}" aria-disabled="${x => x.ariaDisabled}" aria-errormessage="${x => x.ariaErrormessage}" aria-flowto="${x => x.ariaFlowto}" aria-haspopup="${x => x.ariaHaspopup}" aria-hidden="${x => x.ariaHidden}" aria-invalid="${x => x.ariaInvalid}" aria-keyshortcuts="${x => x.ariaKeyshortcuts}" aria-label="${x => x.ariaLabel}" aria-labelledby="${x => x.ariaLabelledby}" aria-live="${x => x.ariaLive}" aria-owns="${x => x.ariaOwns}" aria-relevant="${x => x.ariaRelevant}" aria-roledescription="${x => x.ariaRoledescription}" ${ref("control")} /><slot name="close-button"><button class="clear-button ${x => x.value ? "" : "clear-button__hidden"}" part="clear-button" tabindex="-1" @click=${x => x.handleClearInput()}><slot name="close-glyph"><svg width="9" height="9" viewBox="0 0 9 9" xmlns="http://www.w3.org/2000/svg"><path d="M0.146447 0.146447C0.338683 -0.0478972 0.645911 -0.0270359 0.853553 0.146447L4.5 3.793L8.14645 0.146447C8.34171 -0.0488155 8.65829 -0.0488155 8.85355 0.146447C9.04882 0.341709 9.04882 0.658291 8.85355 0.853553L5.207 4.5L8.85355 8.14645C9.05934 8.35223 9.03129 8.67582 8.85355 8.85355C8.67582 9.03129 8.35409 9.02703 8.14645 8.85355L4.5 5.207L0.853553 8.85355C0.658291 9.04882 0.341709 9.04882 0.146447 8.85355C-0.0488155 8.65829 -0.0488155 8.34171 0.146447 8.14645L3.793 4.5L0.146447 0.853553C-0.0268697 0.680237 -0.0457894 0.34079 0.146447 0.146447Z" /></svg></slot></button></slot></div>${endSlotTemplate(context, definition)}</div></template>`;

class _Search extends FoundationElement {}
/**
 * A form-associated base class for the {@link @microsoft/fast-foundation#(Search:class)} component.
 *
 * @internal
 */


class FormAssociatedSearch extends FormAssociated(_Search) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("input");
  }

}

/**
 * A Search Custom HTML Element.
 * Based largely on the {@link https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/search | <input type="search" /> element }.
 *
 * @slot start - Content which can be provided before the search input
 * @slot end - Content which can be provided after the search clear button
 * @slot - The default slot for the label
 * @slot close-button - The clear button
 * @slot close-glyph - The clear glyph
 * @csspart label - The label
 * @csspart root - The element wrapping the control, including start and end slots
 * @csspart control - The element representing the input
 * @csspart clear-button - The button to clear the input
 *
 * @public
 */

class Search$1 extends FormAssociatedSearch {
  readOnlyChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.readOnly = this.readOnly;
      this.validate();
    }
  }

  autofocusChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.autofocus = this.autofocus;
      this.validate();
    }
  }

  placeholderChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.placeholder = this.placeholder;
    }
  }

  listChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.setAttribute("list", this.list);
      this.validate();
    }
  }

  maxlengthChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.maxLength = this.maxlength;
      this.validate();
    }
  }

  minlengthChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.minLength = this.minlength;
      this.validate();
    }
  }

  patternChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.pattern = this.pattern;
      this.validate();
    }
  }

  sizeChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.size = this.size;
    }
  }

  spellcheckChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.spellcheck = this.spellcheck;
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.validate();

    if (this.autofocus) {
      DOM.queueUpdate(() => {
        this.focus();
      });
    }
  }
  /**
   * Handles the internal control's `input` event
   * @internal
   */


  handleTextInput() {
    this.value = this.control.value;
  }
  /**
   * Handles the control's clear value event
   * @public
   */


  handleClearInput() {
    this.value = "";
    this.control.focus();
  }
  /**
   * Change event handler for inner control.
   * @remarks
   * "Change" events are not `composable` so they will not
   * permeate the shadow DOM boundary. This fn effectively proxies
   * the change event, emitting a `change` event whenever the internal
   * control emits a `change` event
   * @internal
   */


  handleChange() {
    this.$emit("change");
  }

}

__decorate$1([attr({
  attribute: "readonly",
  mode: "boolean"
})], Search$1.prototype, "readOnly", void 0);

__decorate$1([attr({
  mode: "boolean"
})], Search$1.prototype, "autofocus", void 0);

__decorate$1([attr], Search$1.prototype, "placeholder", void 0);

__decorate$1([attr], Search$1.prototype, "list", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], Search$1.prototype, "maxlength", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], Search$1.prototype, "minlength", void 0);

__decorate$1([attr], Search$1.prototype, "pattern", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], Search$1.prototype, "size", void 0);

__decorate$1([attr({
  mode: "boolean"
})], Search$1.prototype, "spellcheck", void 0);

__decorate$1([observable], Search$1.prototype, "defaultSlottedNodes", void 0);
/**
 * Includes ARIA states and properties relating to the ARIA textbox role
 *
 * @public
 */


class DelegatesARIASearch {}
applyMixins(DelegatesARIASearch, ARIAGlobalStatesAndProperties);
applyMixins(Search$1, StartEnd, DelegatesARIASearch);

class _Select extends ListboxElement {}
/**
 * A form-associated base class for the {@link @microsoft/fast-foundation#(Select:class)} component.
 *
 * @internal
 */


class FormAssociatedSelect extends FormAssociated(_Select) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("select");
  }

}

/**
 * A Select Custom HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#select | ARIA select }.
 *
 * @slot start - Content which can be provided before the button content
 * @slot end - Content which can be provided after the button content
 * @slot button-container - The element representing the select button
 * @slot selected-value - The selected value
 * @slot indicator - The visual indicator for the expand/collapse state of the button
 * @slot - The default slot for slotted options
 * @csspart control - The element representing the select invoking element
 * @csspart selected-value - The element wrapping the selected value
 * @csspart indicator - The element wrapping the visual indicator
 * @csspart listbox - The listbox element
 * @fires input - Fires a custom 'input' event when the value updates
 * @fires change - Fires a custom 'change' event when the value updates
 *
 * @public
 */

class Select$1 extends FormAssociatedSelect {
  constructor() {
    super(...arguments);
    /**
     * The open attribute.
     *
     * @public
     * @remarks
     * HTML Attribute: open
     */

    this.open = false;
    /**
     * Indicates the initial state of the position attribute.
     *
     * @internal
     */

    this.forcedPosition = false;
    /**
     * The unique id for the internal listbox element.
     *
     * @internal
     */

    this.listboxId = uniqueId("listbox-");
    /**
     * The max height for the listbox when opened.
     *
     * @internal
     */

    this.maxHeight = 0;
  }
  /**
   * Sets focus and synchronizes ARIA attributes when the open property changes.
   *
   * @param prev - the previous open value
   * @param next - the current open value
   *
   * @internal
   */


  openChanged(prev, next) {
    if (!this.collapsible) {
      return;
    }

    if (this.open) {
      this.ariaControls = this.listboxId;
      this.ariaExpanded = "true";
      this.setPositioning();
      this.focusAndScrollOptionIntoView();
      this.indexWhenOpened = this.selectedIndex; // focus is directed to the element when `open` is changed programmatically

      DOM.queueUpdate(() => this.focus());
      return;
    }

    this.ariaControls = "";
    this.ariaExpanded = "false";
  }
  /**
   * The component is collapsible when in single-selection mode with no size attribute.
   *
   * @internal
   */


  get collapsible() {
    return !(this.multiple || typeof this.size === "number");
  }
  /**
   * The value property.
   *
   * @public
   */


  get value() {
    Observable.track(this, "value");
    return this._value;
  }

  set value(next) {
    var _a, _b, _c, _d, _e, _f, _g;

    const prev = `${this._value}`;

    if ((_a = this._options) === null || _a === void 0 ? void 0 : _a.length) {
      const selectedIndex = this._options.findIndex(el => el.value === next);

      const prevSelectedValue = (_c = (_b = this._options[this.selectedIndex]) === null || _b === void 0 ? void 0 : _b.value) !== null && _c !== void 0 ? _c : null;
      const nextSelectedValue = (_e = (_d = this._options[selectedIndex]) === null || _d === void 0 ? void 0 : _d.value) !== null && _e !== void 0 ? _e : null;

      if (selectedIndex === -1 || prevSelectedValue !== nextSelectedValue) {
        next = "";
        this.selectedIndex = selectedIndex;
      }

      next = (_g = (_f = this.firstSelectedOption) === null || _f === void 0 ? void 0 : _f.value) !== null && _g !== void 0 ? _g : next;
    }

    if (prev !== next) {
      this._value = next;
      super.valueChanged(prev, next);
      Observable.notify(this, "value");
      this.updateDisplayValue();
    }
  }
  /**
   * Sets the value and display value to match the first selected option.
   *
   * @param shouldEmit - if true, the input and change events will be emitted
   *
   * @internal
   */


  updateValue(shouldEmit) {
    var _a, _b;

    if (this.$fastController.isConnected) {
      this.value = (_b = (_a = this.firstSelectedOption) === null || _a === void 0 ? void 0 : _a.value) !== null && _b !== void 0 ? _b : "";
    }

    if (shouldEmit) {
      this.$emit("input");
      this.$emit("change", this, {
        bubbles: true,
        composed: undefined
      });
    }
  }
  /**
   * Updates the proxy value when the selected index changes.
   *
   * @param prev - the previous selected index
   * @param next - the next selected index
   *
   * @internal
   */


  selectedIndexChanged(prev, next) {
    super.selectedIndexChanged(prev, next);
    this.updateValue();
  }

  positionChanged(prev, next) {
    this.positionAttribute = next;
    this.setPositioning();
  }
  /**
   * Calculate and apply listbox positioning based on available viewport space.
   *
   * @public
   */


  setPositioning() {
    const currentBox = this.getBoundingClientRect();
    const viewportHeight = window.innerHeight;
    const availableBottom = viewportHeight - currentBox.bottom;
    this.position = this.forcedPosition ? this.positionAttribute : currentBox.top > availableBottom ? SelectPosition.above : SelectPosition.below;
    this.positionAttribute = this.forcedPosition ? this.positionAttribute : this.position;
    this.maxHeight = this.position === SelectPosition.above ? ~~currentBox.top : ~~availableBottom;
  }
  /**
   * The value displayed on the button.
   *
   * @public
   */


  get displayValue() {
    var _a, _b;

    Observable.track(this, "displayValue");
    return (_b = (_a = this.firstSelectedOption) === null || _a === void 0 ? void 0 : _a.text) !== null && _b !== void 0 ? _b : "";
  }
  /**
   * Synchronize the `aria-disabled` property when the `disabled` property changes.
   *
   * @param prev - The previous disabled value
   * @param next - The next disabled value
   *
   * @internal
   */


  disabledChanged(prev, next) {
    if (super.disabledChanged) {
      super.disabledChanged(prev, next);
    }

    this.ariaDisabled = this.disabled ? "true" : "false";
  }
  /**
   * Reset the element to its first selectable option when its parent form is reset.
   *
   * @internal
   */


  formResetCallback() {
    this.setProxyOptions(); // Call the base class's implementation setDefaultSelectedOption instead of the select's
    // override, in order to reset the selectedIndex without using the value property.

    super.setDefaultSelectedOption();

    if (this.selectedIndex === -1) {
      this.selectedIndex = 0;
    }
  }
  /**
   * Handle opening and closing the listbox when the select is clicked.
   *
   * @param e - the mouse event
   * @internal
   */


  clickHandler(e) {
    // do nothing if the select is disabled
    if (this.disabled) {
      return;
    }

    if (this.open) {
      const captured = e.target.closest(`option,[role=option]`);

      if (captured && captured.disabled) {
        return;
      }
    }

    super.clickHandler(e);
    this.open = this.collapsible && !this.open;

    if (!this.open && this.indexWhenOpened !== this.selectedIndex) {
      this.updateValue(true);
    }

    return true;
  }
  /**
   * Handles focus state when the element or its children lose focus.
   *
   * @param e - The focus event
   * @internal
   */


  focusoutHandler(e) {
    var _a;

    super.focusoutHandler(e);

    if (!this.open) {
      return true;
    }

    const focusTarget = e.relatedTarget;

    if (this.isSameNode(focusTarget)) {
      this.focus();
      return;
    }

    if (!((_a = this.options) === null || _a === void 0 ? void 0 : _a.includes(focusTarget))) {
      this.open = false;

      if (this.indexWhenOpened !== this.selectedIndex) {
        this.updateValue(true);
      }
    }
  }
  /**
   * Updates the value when an option's value changes.
   *
   * @param source - the source object
   * @param propertyName - the property to evaluate
   *
   * @internal
   * @override
   */


  handleChange(source, propertyName) {
    super.handleChange(source, propertyName);

    if (propertyName === "value") {
      this.updateValue();
    }
  }
  /**
   * Synchronize the form-associated proxy and updates the value property of the element.
   *
   * @param prev - the previous collection of slotted option elements
   * @param next - the next collection of slotted option elements
   *
   * @internal
   */


  slottedOptionsChanged(prev, next) {
    this.options.forEach(o => {
      const notifier = Observable.getNotifier(o);
      notifier.unsubscribe(this, "value");
    });
    super.slottedOptionsChanged(prev, next);
    this.options.forEach(o => {
      const notifier = Observable.getNotifier(o);
      notifier.subscribe(this, "value");
    });
    this.setProxyOptions();
    this.updateValue();
  }
  /**
   * Prevents focus when size is set and a scrollbar is clicked.
   *
   * @param e - the mouse event object
   *
   * @override
   * @internal
   */


  mousedownHandler(e) {
    var _a;

    if (e.offsetX >= 0 && e.offsetX <= ((_a = this.listbox) === null || _a === void 0 ? void 0 : _a.scrollWidth)) {
      return super.mousedownHandler(e);
    }

    return this.collapsible;
  }
  /**
   * Sets the multiple property on the proxy element.
   *
   * @param prev - the previous multiple value
   * @param next - the current multiple value
   */


  multipleChanged(prev, next) {
    super.multipleChanged(prev, next);

    if (this.proxy) {
      this.proxy.multiple = next;
    }
  }
  /**
   * Updates the selectedness of each option when the list of selected options changes.
   *
   * @param prev - the previous list of selected options
   * @param next - the current list of selected options
   *
   * @override
   * @internal
   */


  selectedOptionsChanged(prev, next) {
    var _a;

    super.selectedOptionsChanged(prev, next);
    (_a = this.options) === null || _a === void 0 ? void 0 : _a.forEach((o, i) => {
      var _a;

      const proxyOption = (_a = this.proxy) === null || _a === void 0 ? void 0 : _a.options.item(i);

      if (proxyOption) {
        proxyOption.selected = o.selected;
      }
    });
  }
  /**
   * Sets the selected index to match the first option with the selected attribute, or
   * the first selectable option.
   *
   * @override
   * @internal
   */


  setDefaultSelectedOption() {
    var _a;

    const options = (_a = this.options) !== null && _a !== void 0 ? _a : Array.from(this.children).filter(Listbox$1.slottedOptionFilter);
    const selectedIndex = options === null || options === void 0 ? void 0 : options.findIndex(el => el.hasAttribute("selected") || el.selected || el.value === this.value);

    if (selectedIndex !== -1) {
      this.selectedIndex = selectedIndex;
      return;
    }

    this.selectedIndex = 0;
  }
  /**
   * Resets and fills the proxy to match the component's options.
   *
   * @internal
   */


  setProxyOptions() {
    if (this.proxy instanceof HTMLSelectElement && this.options) {
      this.proxy.options.length = 0;
      this.options.forEach(option => {
        const proxyOption = option.proxy || (option instanceof HTMLOptionElement ? option.cloneNode() : null);

        if (proxyOption) {
          this.proxy.options.add(proxyOption);
        }
      });
    }
  }
  /**
   * Handle keyboard interaction for the select.
   *
   * @param e - the keyboard event
   * @internal
   */


  keydownHandler(e) {
    super.keydownHandler(e);
    const key = e.key || e.key.charCodeAt(0);

    switch (key) {
      case keySpace:
        {
          e.preventDefault();

          if (this.collapsible && this.typeAheadExpired) {
            this.open = !this.open;
          }

          break;
        }

      case keyHome:
      case keyEnd:
        {
          e.preventDefault();
          break;
        }

      case keyEnter:
        {
          e.preventDefault();
          this.open = !this.open;
          break;
        }

      case keyEscape:
        {
          if (this.collapsible && this.open) {
            e.preventDefault();
            this.open = false;
          }

          break;
        }

      case keyTab:
        {
          if (this.collapsible && this.open) {
            e.preventDefault();
            this.open = false;
          }

          return true;
        }
    }

    if (!this.open && this.indexWhenOpened !== this.selectedIndex) {
      this.updateValue(true);
      this.indexWhenOpened = this.selectedIndex;
    }

    return !(key in ArrowKeys);
  }

  connectedCallback() {
    super.connectedCallback();
    this.forcedPosition = !!this.positionAttribute;
    this.addEventListener("contentchange", this.updateDisplayValue);
  }

  disconnectedCallback() {
    this.removeEventListener("contentchange", this.updateDisplayValue);
    super.disconnectedCallback();
  }
  /**
   * Updates the proxy's size property when the size attribute changes.
   *
   * @param prev - the previous size
   * @param next - the current size
   *
   * @override
   * @internal
   */


  sizeChanged(prev, next) {
    super.sizeChanged(prev, next);

    if (this.proxy) {
      this.proxy.size = next;
    }
  }
  /**
   *
   * @internal
   */


  updateDisplayValue() {
    if (this.collapsible) {
      Observable.notify(this, "displayValue");
    }
  }

}

__decorate$1([attr({
  attribute: "open",
  mode: "boolean"
})], Select$1.prototype, "open", void 0);

__decorate$1([volatile], Select$1.prototype, "collapsible", null);

__decorate$1([observable], Select$1.prototype, "control", void 0);

__decorate$1([attr({
  attribute: "position"
})], Select$1.prototype, "positionAttribute", void 0);

__decorate$1([observable], Select$1.prototype, "position", void 0);

__decorate$1([observable], Select$1.prototype, "maxHeight", void 0);
/**
 * Includes ARIA states and properties relating to the ARIA select role.
 *
 * @public
 */


class DelegatesARIASelect {}

__decorate$1([observable], DelegatesARIASelect.prototype, "ariaControls", void 0);

applyMixins(DelegatesARIASelect, DelegatesARIAListbox);
applyMixins(Select$1, StartEnd, DelegatesARIASelect);

/**
 * The template for the {@link @microsoft/fast-foundation#(Select:class)} component.
 * @public
 */

const selectTemplate = (context, definition) => html`<template class="${x => [x.collapsible && "collapsible", x.collapsible && x.open && "open", x.disabled && "disabled", x.collapsible && x.position].filter(Boolean).join(" ")}" aria-activedescendant="${x => x.ariaActiveDescendant}" aria-controls="${x => x.ariaControls}" aria-disabled="${x => x.ariaDisabled}" aria-expanded="${x => x.ariaExpanded}" aria-haspopup="${x => x.collapsible ? "listbox" : null}" aria-multiselectable="${x => x.ariaMultiSelectable}" ?open="${x => x.open}" role="combobox" tabindex="${x => !x.disabled ? "0" : null}" @click="${(x, c) => x.clickHandler(c.event)}" @focusin="${(x, c) => x.focusinHandler(c.event)}" @focusout="${(x, c) => x.focusoutHandler(c.event)}" @keydown="${(x, c) => x.keydownHandler(c.event)}" @mousedown="${(x, c) => x.mousedownHandler(c.event)}">${when(x => x.collapsible, html`<div class="control" part="control" ?disabled="${x => x.disabled}" ${ref("control")}>${startSlotTemplate(context, definition)}<slot name="button-container"><div class="selected-value" part="selected-value"><slot name="selected-value">${x => x.displayValue}</slot></div><div aria-hidden="true" class="indicator" part="indicator"><slot name="indicator">${definition.indicator || ""}</slot></div></slot>${endSlotTemplate(context, definition)}</div>`)}<div class="listbox" id="${x => x.listboxId}" part="listbox" role="listbox" ?disabled="${x => x.disabled}" ?hidden="${x => x.collapsible ? !x.open : false}" ${ref("listbox")}><slot ${slotted({
  filter: Listbox$1.slottedOptionFilter,
  flatten: true,
  property: "slottedOptions"
})}></slot></div></template>`;

/**
 * The template for the fast-skeleton component
 * @public
 */

const skeletonTemplate = (context, definition) => html`<template class="${x => x.shape === "circle" ? "circle" : "rect"}" pattern="${x => x.pattern}" ?shimmer="${x => x.shimmer}">${when(x => x.shimmer === true, html`<span class="shimmer"></span>`)}<object type="image/svg+xml" data="${x => x.pattern}" role="presentation"><img class="pattern" src="${x => x.pattern}" /></object><slot></slot></template>`;

/**
 * A Skeleton Custom HTML Element.
 *
 * @slot - The default slot
 *
 * @public
 */

class Skeleton extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * Indicates what the shape of the Skeleton should be.
     *
     * @public
     * @remarks
     * HTML Attribute: shape
     */

    this.shape = "rect";
  }

}

__decorate$1([attr], Skeleton.prototype, "fill", void 0);

__decorate$1([attr], Skeleton.prototype, "shape", void 0);

__decorate$1([attr], Skeleton.prototype, "pattern", void 0);

__decorate$1([attr({
  mode: "boolean"
})], Skeleton.prototype, "shimmer", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(SliderLabel:class)} component.
 * @public
 */

const sliderLabelTemplate = (context, definition) => html`<template aria-disabled="${x => x.disabled}" class="${x => x.sliderOrientation || Orientation.horizontal} ${x => x.disabled ? "disabled" : ""}"><div ${ref("root")} part="root" class="root" style="${x => x.positionStyle}"><div class="container">${when(x => !x.hideMark, html`<div class="mark"></div>`)}<div class="label"><slot></slot></div></div></div></template>`;

/**
 * Converts a pixel coordinate on the track to a percent of the track's range
 */

function convertPixelToPercent(pixelPos, minPosition, maxPosition, direction) {
  let pct = limit(0, 1, (pixelPos - minPosition) / (maxPosition - minPosition));

  if (direction === Direction.rtl) {
    pct = 1 - pct;
  }

  return pct;
}

const defaultConfig = {
  min: 0,
  max: 0,
  direction: Direction.ltr,
  orientation: Orientation.horizontal,
  disabled: false
};
/**
 * A label element intended to be used with the {@link @microsoft/fast-foundation#(Slider:class)} component.
 *
 * @slot - The default slot for the label content
 * @csspart root - The element wrapping the label mark and text
 *
 * @public
 */

class SliderLabel$1 extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * Hides the tick mark.
     *
     * @public
     * @remarks
     * HTML Attribute: hide-mark
     */

    this.hideMark = false;
    /**
     * @internal
     */

    this.sliderDirection = Direction.ltr;

    this.getSliderConfiguration = () => {
      if (!this.isSliderConfig(this.parentNode)) {
        this.sliderDirection = defaultConfig.direction || Direction.ltr;
        this.sliderOrientation = defaultConfig.orientation ;
        this.sliderMaxPosition = defaultConfig.max;
        this.sliderMinPosition = defaultConfig.min;
      } else {
        const parentSlider = this.parentNode;
        const {
          min,
          max,
          direction,
          orientation,
          disabled
        } = parentSlider;

        if (disabled !== undefined) {
          this.disabled = disabled;
        }

        this.sliderDirection = direction || Direction.ltr;
        this.sliderOrientation = orientation || Orientation.horizontal;
        this.sliderMaxPosition = max;
        this.sliderMinPosition = min;
      }
    };

    this.positionAsStyle = () => {
      const direction = this.sliderDirection ? this.sliderDirection : Direction.ltr;
      const pct = convertPixelToPercent(Number(this.position), Number(this.sliderMinPosition), Number(this.sliderMaxPosition));
      let rightNum = Math.round((1 - pct) * 100);
      let leftNum = Math.round(pct * 100);

      if (Number.isNaN(leftNum) && Number.isNaN(rightNum)) {
        rightNum = 50;
        leftNum = 50;
      }

      if (this.sliderOrientation === Orientation.horizontal) {
        return direction === Direction.rtl ? `right: ${leftNum}%; left: ${rightNum}%;` : `left: ${leftNum}%; right: ${rightNum}%;`;
      } else {
        return `top: ${leftNum}%; bottom: ${rightNum}%;`;
      }
    };
  }

  positionChanged() {
    this.positionStyle = this.positionAsStyle();
  }
  /**
   * @internal
   */


  sliderOrientationChanged() {
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.getSliderConfiguration();
    this.positionStyle = this.positionAsStyle();
    this.notifier = Observable.getNotifier(this.parentNode);
    this.notifier.subscribe(this, "orientation");
    this.notifier.subscribe(this, "direction");
    this.notifier.subscribe(this, "max");
    this.notifier.subscribe(this, "min");
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    super.disconnectedCallback();
    this.notifier.unsubscribe(this, "orientation");
    this.notifier.unsubscribe(this, "direction");
    this.notifier.unsubscribe(this, "max");
    this.notifier.unsubscribe(this, "min");
  }
  /**
   * @internal
   */


  handleChange(source, propertyName) {
    switch (propertyName) {
      case "direction":
        this.sliderDirection = source.direction;
        break;

      case "orientation":
        this.sliderOrientation = source.orientation;
        break;

      case "max":
        this.sliderMinPosition = source.max;
        break;

      case "min":
        this.sliderMinPosition = source.min;
        break;
    }

    this.positionStyle = this.positionAsStyle();
  }

  isSliderConfig(node) {
    return node.max !== undefined && node.min !== undefined;
  }

}

__decorate$1([observable], SliderLabel$1.prototype, "positionStyle", void 0);

__decorate$1([attr], SliderLabel$1.prototype, "position", void 0);

__decorate$1([attr({
  attribute: "hide-mark",
  mode: "boolean"
})], SliderLabel$1.prototype, "hideMark", void 0);

__decorate$1([attr({
  attribute: "disabled",
  mode: "boolean"
})], SliderLabel$1.prototype, "disabled", void 0);

__decorate$1([observable], SliderLabel$1.prototype, "sliderOrientation", void 0);

__decorate$1([observable], SliderLabel$1.prototype, "sliderMinPosition", void 0);

__decorate$1([observable], SliderLabel$1.prototype, "sliderMaxPosition", void 0);

__decorate$1([observable], SliderLabel$1.prototype, "sliderDirection", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(Slider:class)} component.
 * @public
 */

const sliderTemplate = (context, definition) => html`<template role="slider" class="${x => x.readOnly ? "readonly" : ""} ${x => x.orientation || Orientation.horizontal}" tabindex="${x => x.disabled ? null : 0}" aria-valuetext="${x => x.valueTextFormatter(x.value)}" aria-valuenow="${x => x.value}" aria-valuemin="${x => x.min}" aria-valuemax="${x => x.max}" aria-disabled="${x => x.disabled ? true : void 0}" aria-readonly="${x => x.readOnly ? true : void 0}" aria-orientation="${x => x.orientation}" class="${x => x.orientation}"><div part="positioning-region" class="positioning-region"><div ${ref("track")} part="track-container" class="track"><slot name="track"></slot><div part="track-start" class="track-start" style="${x => x.position}"><slot name="track-start"></slot></div></div><slot></slot><div ${ref("thumb")} part="thumb-container" class="thumb-container" style="${x => x.position}"><slot name="thumb">${definition.thumb || ""}</slot></div></div></template>`;

class _Slider extends FoundationElement {}
/**
 * A form-associated base class for the {@link @microsoft/fast-foundation#(Slider:class)} component.
 *
 * @internal
 */


class FormAssociatedSlider extends FormAssociated(_Slider) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("input");
  }

}

/**
 * The selection modes of a {@link @microsoft/fast-foundation#(Slider:class)}.
 * @public
 */

const SliderMode = {
  singleValue: "single-value"
};
/**
 * A Slider Custom HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#slider | ARIA slider }.
 *
 * @slot track - The track of the slider
 * @slot track-start - The track-start visual indicator
 * @slot thumb - The slider thumb
 * @slot - The default slot for labels
 * @csspart positioning-region - The region used to position the elements of the slider
 * @csspart track-container - The region containing the track elements
 * @csspart track-start - The element wrapping the track start slot
 * @csspart thumb-container - The thumb container element which is programatically positioned
 * @fires change - Fires a custom 'change' event when the slider value changes
 *
 * @public
 */

class Slider extends FormAssociatedSlider {
  constructor() {
    super(...arguments);
    /**
     * @internal
     */

    this.direction = Direction.ltr;
    /**
     * @internal
     */

    this.isDragging = false;
    /**
     * @internal
     */

    this.trackWidth = 0;
    /**
     * @internal
     */

    this.trackMinWidth = 0;
    /**
     * @internal
     */

    this.trackHeight = 0;
    /**
     * @internal
     */

    this.trackLeft = 0;
    /**
     * @internal
     */

    this.trackMinHeight = 0;
    /**
     * Custom function that generates a string for the component's "aria-valuetext" attribute based on the current value.
     *
     * @public
     */

    this.valueTextFormatter = () => null;
    /**
     * The minimum allowed value.
     *
     * @defaultValue - 0
     * @public
     * @remarks
     * HTML Attribute: min
     */


    this.min = 0; // Map to proxy element.

    /**
     * The maximum allowed value.
     *
     * @defaultValue - 10
     * @public
     * @remarks
     * HTML Attribute: max
     */

    this.max = 10; // Map to proxy element.

    /**
     * Value to increment or decrement via arrow keys, mouse click or drag.
     *
     * @public
     * @remarks
     * HTML Attribute: step
     */

    this.step = 1; // Map to proxy element.

    /**
     * The orientation of the slider.
     *
     * @public
     * @remarks
     * HTML Attribute: orientation
     */

    this.orientation = Orientation.horizontal;
    /**
     * The selection mode.
     *
     * @public
     * @remarks
     * HTML Attribute: mode
     */

    this.mode = SliderMode.singleValue;

    this.keypressHandler = e => {
      if (e.key === keyHome) {
        e.preventDefault();
        this.value = `${this.min}`;
      } else if (e.key === keyEnd) {
        e.preventDefault();
        this.value = `${this.max}`;
      } else if (!e.shiftKey) {
        switch (e.key) {
          case keyArrowRight:
          case keyArrowUp:
            e.preventDefault();
            this.increment();
            break;

          case keyArrowLeft:
          case keyArrowDown:
            e.preventDefault();
            this.decrement();
            break;
        }
      }
    };

    this.setupTrackConstraints = () => {
      const clientRect = this.track.getBoundingClientRect();
      this.trackWidth = this.track.clientWidth;
      this.trackMinWidth = this.track.clientLeft;
      this.trackHeight = clientRect.bottom;
      this.trackMinHeight = clientRect.top;
      this.trackLeft = this.getBoundingClientRect().left;

      if (this.trackWidth === 0) {
        this.trackWidth = 1;
      }
    };

    this.setupListeners = (remove = false) => {
      const eventAction = `${remove ? "remove" : "add"}EventListener`;
      this[eventAction]("keydown", this.keypressHandler);
      this[eventAction]("mousedown", this.handleMouseDown);
      this.thumb[eventAction]("mousedown", this.handleThumbMouseDown, {
        passive: true
      });
      this.thumb[eventAction]("touchstart", this.handleThumbMouseDown, {
        passive: true
      }); // removes handlers attached by mousedown handlers

      if (remove) {
        this.handleMouseDown(null);
        this.handleThumbMouseDown(null);
      }
    };
    /**
     * @internal
     */


    this.initialValue = "";
    /**
     *  Handle mouse moves during a thumb drag operation
     *  If the event handler is null it removes the events
     */

    this.handleThumbMouseDown = event => {
      if (event) {
        if (this.readOnly || this.disabled || event.defaultPrevented) {
          return;
        }

        event.target.focus();
      }

      const eventAction = `${event !== null ? "add" : "remove"}EventListener`;
      window[eventAction]("mouseup", this.handleWindowMouseUp);
      window[eventAction]("mousemove", this.handleMouseMove, {
        passive: true
      });
      window[eventAction]("touchmove", this.handleMouseMove, {
        passive: true
      });
      window[eventAction]("touchend", this.handleWindowMouseUp);
      this.isDragging = event !== null;
    };
    /**
     *  Handle mouse moves during a thumb drag operation
     */


    this.handleMouseMove = e => {
      if (this.readOnly || this.disabled || e.defaultPrevented) {
        return;
      } // update the value based on current position


      const sourceEvent = window.TouchEvent && e instanceof TouchEvent ? e.touches[0] : e;
      const eventValue = this.orientation === Orientation.horizontal ? sourceEvent.pageX - document.documentElement.scrollLeft - this.trackLeft : sourceEvent.pageY - document.documentElement.scrollTop;
      this.value = `${this.calculateNewValue(eventValue)}`;
    };

    this.calculateNewValue = rawValue => {
      // update the value based on current position
      const newPosition = convertPixelToPercent(rawValue, this.orientation === Orientation.horizontal ? this.trackMinWidth : this.trackMinHeight, this.orientation === Orientation.horizontal ? this.trackWidth : this.trackHeight, this.direction);
      const newValue = (this.max - this.min) * newPosition + this.min;
      return this.convertToConstrainedValue(newValue);
    };
    /**
     * Handle a window mouse up during a drag operation
     */


    this.handleWindowMouseUp = event => {
      this.stopDragging();
    };

    this.stopDragging = () => {
      this.isDragging = false;
      this.handleMouseDown(null);
      this.handleThumbMouseDown(null);
    };
    /**
     *
     * @param e - MouseEvent or null. If there is no event handler it will remove the events
     */


    this.handleMouseDown = e => {
      const eventAction = `${e !== null ? "add" : "remove"}EventListener`;

      if (e === null || !this.disabled && !this.readOnly) {
        window[eventAction]("mouseup", this.handleWindowMouseUp);
        window.document[eventAction]("mouseleave", this.handleWindowMouseUp);
        window[eventAction]("mousemove", this.handleMouseMove);

        if (e) {
          e.preventDefault();
          this.setupTrackConstraints();
          e.target.focus();
          const controlValue = this.orientation === Orientation.horizontal ? e.pageX - document.documentElement.scrollLeft - this.trackLeft : e.pageY - document.documentElement.scrollTop;
          this.value = `${this.calculateNewValue(controlValue)}`;
        }
      }
    };

    this.convertToConstrainedValue = value => {
      if (isNaN(value)) {
        value = this.min;
      }
      /**
       * The following logic intends to overcome the issue with math in JavaScript with regards to floating point numbers.
       * This is needed as the `step` may be an integer but could also be a float. To accomplish this the step  is assumed to be a float
       * and is converted to an integer by determining the number of decimal places it represent, multiplying it until it is an
       * integer and then dividing it to get back to the correct number.
       */


      let constrainedValue = value - this.min;
      const roundedConstrainedValue = Math.round(constrainedValue / this.step);
      const remainderValue = constrainedValue - roundedConstrainedValue * (this.stepMultiplier * this.step) / this.stepMultiplier;
      constrainedValue = remainderValue >= Number(this.step) / 2 ? constrainedValue - remainderValue + Number(this.step) : constrainedValue - remainderValue;
      return constrainedValue + this.min;
    };
  }

  readOnlyChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.readOnly = this.readOnly;
    }
  }
  /**
   * The value property, typed as a number.
   *
   * @public
   */


  get valueAsNumber() {
    return parseFloat(super.value);
  }

  set valueAsNumber(next) {
    this.value = next.toString();
  }
  /**
   * @internal
   */


  valueChanged(previous, next) {
    super.valueChanged(previous, next);

    if (this.$fastController.isConnected) {
      this.setThumbPositionForOrientation(this.direction);
    }

    this.$emit("change");
  }

  minChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.min = `${this.min}`;
    }

    this.validate();
  }

  maxChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.max = `${this.max}`;
    }

    this.validate();
  }

  stepChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.step = `${this.step}`;
    }

    this.updateStepMultiplier();
    this.validate();
  }

  orientationChanged() {
    if (this.$fastController.isConnected) {
      this.setThumbPositionForOrientation(this.direction);
    }
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.proxy.setAttribute("type", "range");
    this.direction = getDirection(this);
    this.updateStepMultiplier();
    this.setupTrackConstraints();
    this.setupListeners();
    this.setupDefaultValue();
    this.setThumbPositionForOrientation(this.direction);
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    this.setupListeners(true);
  }
  /**
   * Increment the value by the step
   *
   * @public
   */


  increment() {
    const newVal = this.direction !== Direction.rtl && this.orientation !== Orientation.vertical ? Number(this.value) + Number(this.step) : Number(this.value) - Number(this.step);
    const incrementedVal = this.convertToConstrainedValue(newVal);
    const incrementedValString = incrementedVal < Number(this.max) ? `${incrementedVal}` : `${this.max}`;
    this.value = incrementedValString;
  }
  /**
   * Decrement the value by the step
   *
   * @public
   */


  decrement() {
    const newVal = this.direction !== Direction.rtl && this.orientation !== Orientation.vertical ? Number(this.value) - Number(this.step) : Number(this.value) + Number(this.step);
    const decrementedVal = this.convertToConstrainedValue(newVal);
    const decrementedValString = decrementedVal > Number(this.min) ? `${decrementedVal}` : `${this.min}`;
    this.value = decrementedValString;
  }
  /**
   * Places the thumb based on the current value
   *
   * @public
   * @param direction - writing mode
   */


  setThumbPositionForOrientation(direction) {
    const newPct = convertPixelToPercent(Number(this.value), Number(this.min), Number(this.max), direction);
    const percentage = (1 - newPct) * 100;

    if (this.orientation === Orientation.horizontal) {
      this.position = this.isDragging ? `right: ${percentage}%; transition: none;` : `right: ${percentage}%; transition: all 0.2s ease;`;
    } else {
      this.position = this.isDragging ? `bottom: ${percentage}%; transition: none;` : `bottom: ${percentage}%; transition: all 0.2s ease;`;
    }
  }
  /**
   * Update the step multiplier used to ensure rounding errors from steps that
   * are not whole numbers
   */


  updateStepMultiplier() {
    const stepString = this.step + "";
    const decimalPlacesOfStep = !!(this.step % 1) ? stepString.length - stepString.indexOf(".") - 1 : 0;
    this.stepMultiplier = Math.pow(10, decimalPlacesOfStep);
  }

  get midpoint() {
    return `${this.convertToConstrainedValue((this.max + this.min) / 2)}`;
  }

  setupDefaultValue() {
    if (typeof this.value === "string") {
      if (this.value.length === 0) {
        this.initialValue = this.midpoint;
      } else {
        const value = parseFloat(this.value);

        if (!Number.isNaN(value) && (value < this.min || value > this.max)) {
          this.value = this.midpoint;
        }
      }
    }
  }

}

__decorate$1([attr({
  attribute: "readonly",
  mode: "boolean"
})], Slider.prototype, "readOnly", void 0);

__decorate$1([observable], Slider.prototype, "direction", void 0);

__decorate$1([observable], Slider.prototype, "isDragging", void 0);

__decorate$1([observable], Slider.prototype, "position", void 0);

__decorate$1([observable], Slider.prototype, "trackWidth", void 0);

__decorate$1([observable], Slider.prototype, "trackMinWidth", void 0);

__decorate$1([observable], Slider.prototype, "trackHeight", void 0);

__decorate$1([observable], Slider.prototype, "trackLeft", void 0);

__decorate$1([observable], Slider.prototype, "trackMinHeight", void 0);

__decorate$1([observable], Slider.prototype, "valueTextFormatter", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], Slider.prototype, "min", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], Slider.prototype, "max", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], Slider.prototype, "step", void 0);

__decorate$1([attr], Slider.prototype, "orientation", void 0);

__decorate$1([attr], Slider.prototype, "mode", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(Switch:class)} component.
 * @public
 */

const switchTemplate = (context, definition) => html`<template role="switch" aria-checked="${x => x.checked}" aria-disabled="${x => x.disabled}" aria-readonly="${x => x.readOnly}" tabindex="${x => x.disabled ? null : 0}" @keypress="${(x, c) => x.keypressHandler(c.event)}" @click="${(x, c) => x.clickHandler(c.event)}" class="${x => x.checked ? "checked" : ""}"><label part="label" class="${x => x.defaultSlottedNodes && x.defaultSlottedNodes.length ? "label" : "label label__hidden"}"><slot ${slotted("defaultSlottedNodes")}></slot></label><div part="switch" class="switch"><slot name="switch">${definition.switch || ""}</slot></div><span class="status-message" part="status-message"><span class="checked-message" part="checked-message"><slot name="checked-message"></slot></span><span class="unchecked-message" part="unchecked-message"><slot name="unchecked-message"></slot></span></span></template>`;

class _Switch extends FoundationElement {}
/**
 * A form-associated base class for the {@link @microsoft/fast-foundation#(Switch:class)} component.
 *
 * @internal
 */


class FormAssociatedSwitch extends CheckableFormAssociated(_Switch) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("input");
  }

}

/**
 * A Switch Custom HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#switch | ARIA switch }.
 *
 * @slot - The deafult slot for the label
 * @slot checked-message - The message when in a checked state
 * @slot unchecked-message - The message when in an unchecked state
 * @csspart label - The label
 * @csspart switch - The element representing the switch, which wraps the indicator
 * @csspart status-message - The wrapper for the status messages
 * @csspart checked-message - The checked message
 * @csspart unchecked-message - The unchecked message
 * @fires change - Emits a custom change event when the checked state changes
 *
 * @public
 */

class Switch extends FormAssociatedSwitch {
  constructor() {
    super();
    /**
     * The element's value to be included in form submission when checked.
     * Default to "on" to reach parity with input[type="checkbox"]
     *
     * @internal
     */

    this.initialValue = "on";
    /**
     * @internal
     */

    this.keypressHandler = e => {
      switch (e.key) {
        case keyEnter:
        case keySpace:
          this.checked = !this.checked;
          break;
      }
    };
    /**
     * @internal
     */


    this.clickHandler = e => {
      if (!this.disabled && !this.readOnly) {
        this.checked = !this.checked;
      }
    };

    this.proxy.setAttribute("type", "checkbox");
  }

  readOnlyChanged() {
    if (this.proxy instanceof HTMLInputElement) {
      this.proxy.readOnly = this.readOnly;
    }

    this.readOnly ? this.classList.add("readonly") : this.classList.remove("readonly");
  }
  /**
   * @internal
   */


  checkedChanged(prev, next) {
    super.checkedChanged(prev, next);
    /**
     * @deprecated - this behavior already exists in the template and should not exist in the class.
     */

    this.checked ? this.classList.add("checked") : this.classList.remove("checked");
  }

}

__decorate$1([attr({
  attribute: "readonly",
  mode: "boolean"
})], Switch.prototype, "readOnly", void 0);

__decorate$1([observable], Switch.prototype, "defaultSlottedNodes", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#TabPanel} component.
 * @public
 */

const tabPanelTemplate = (context, definition) => html`<template slot="tabpanel" role="tabpanel"><slot></slot></template>`;

/**
 * A TabPanel Component to be used with {@link @microsoft/fast-foundation#(Tabs:class)}
 *
 * @slot - The default slot for the tabpanel content
 *
 * @public
 */

class TabPanel extends FoundationElement {}

/**
 * The template for the {@link @microsoft/fast-foundation#Tab} component.
 * @public
 */

const tabTemplate = (context, definition) => html`<template slot="tab" role="tab" aria-disabled="${x => x.disabled}"><slot></slot></template>`;

/**
 * A Tab Component to be used with {@link @microsoft/fast-foundation#(Tabs:class)}
 *
 * @slot - The default slot for the tab content
 *
 * @public
 */

class Tab extends FoundationElement {}

__decorate$1([attr({
  mode: "boolean"
})], Tab.prototype, "disabled", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(Tabs:class)} component.
 * @public
 */

const tabsTemplate = (context, definition) => html`<template class="${x => x.orientation}">${startSlotTemplate(context, definition)}<div class="tablist" part="tablist" role="tablist"><slot class="tab" name="tab" part="tab" ${slotted("tabs")}></slot>${when(x => x.showActiveIndicator, html`<div ${ref("activeIndicatorRef")} class="activeIndicator" part="activeIndicator"></div>`)}</div>${endSlotTemplate(context, definition)}<div class="tabpanel"><slot name="tabpanel" part="tabpanel" ${slotted("tabpanels")}></slot></div></template>`;

/**
 * The orientation of the {@link @microsoft/fast-foundation#(Tabs:class)} component
 * @public
 */

const TabsOrientation = {
  vertical: "vertical",
  horizontal: "horizontal"
};
/**
 * A Tabs Custom HTML Element.
 * Implements the {@link https://www.w3.org/TR/wai-aria-1.1/#tablist | ARIA tablist }.
 *
 * @slot start - Content which can be provided before the tablist element
 * @slot end - Content which can be provided after the tablist element
 * @slot tab - The slot for tabs
 * @slot tabpanel - The slot for tabpanels
 * @csspart tablist - The element wrapping for the tabs
 * @csspart tab - The tab slot
 * @csspart activeIndicator - The visual indicator
 * @csspart tabpanel - The tabpanel slot
 * @fires change - Fires a custom 'change' event when a tab is clicked or during keyboard navigation
 *
 * @public
 */

class Tabs extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * The orientation
     * @public
     * @remarks
     * HTML Attribute: orientation
     */

    this.orientation = TabsOrientation.horizontal;
    /**
     * Whether or not to show the active indicator
     * @public
     * @remarks
     * HTML Attribute: activeindicator
     */

    this.activeindicator = true;
    /**
     * @internal
     */

    this.showActiveIndicator = true;
    this.prevActiveTabIndex = 0;
    this.activeTabIndex = 0;
    this.ticking = false;

    this.change = () => {
      this.$emit("change", this.activetab);
    };

    this.isDisabledElement = el => {
      return el.getAttribute("aria-disabled") === "true";
    };

    this.isFocusableElement = el => {
      return !this.isDisabledElement(el);
    };

    this.setTabs = () => {
      const gridHorizontalProperty = "gridColumn";
      const gridVerticalProperty = "gridRow";
      const gridProperty = this.isHorizontal() ? gridHorizontalProperty : gridVerticalProperty;
      this.activeTabIndex = this.getActiveIndex();
      this.showActiveIndicator = false;
      this.tabs.forEach((tab, index) => {
        if (tab.slot === "tab") {
          const isActiveTab = this.activeTabIndex === index && this.isFocusableElement(tab);

          if (this.activeindicator && this.isFocusableElement(tab)) {
            this.showActiveIndicator = true;
          }

          const tabId = this.tabIds[index];
          const tabpanelId = this.tabpanelIds[index];
          tab.setAttribute("id", tabId);
          tab.setAttribute("aria-selected", isActiveTab ? "true" : "false");
          tab.setAttribute("aria-controls", tabpanelId);
          tab.addEventListener("click", this.handleTabClick);
          tab.addEventListener("keydown", this.handleTabKeyDown);
          tab.setAttribute("tabindex", isActiveTab ? "0" : "-1");

          if (isActiveTab) {
            this.activetab = tab;
          }
        } // If the original property isn't emptied out,
        // the next set will morph into a grid-area style setting that is not what we want


        tab.style[gridHorizontalProperty] = "";
        tab.style[gridVerticalProperty] = "";
        tab.style[gridProperty] = `${index + 1}`;
        !this.isHorizontal() ? tab.classList.add("vertical") : tab.classList.remove("vertical");
      });
    };

    this.setTabPanels = () => {
      this.tabpanels.forEach((tabpanel, index) => {
        const tabId = this.tabIds[index];
        const tabpanelId = this.tabpanelIds[index];
        tabpanel.setAttribute("id", tabpanelId);
        tabpanel.setAttribute("aria-labelledby", tabId);
        this.activeTabIndex !== index ? tabpanel.setAttribute("hidden", "") : tabpanel.removeAttribute("hidden");
      });
    };

    this.handleTabClick = event => {
      const selectedTab = event.currentTarget;

      if (selectedTab.nodeType === 1 && this.isFocusableElement(selectedTab)) {
        this.prevActiveTabIndex = this.activeTabIndex;
        this.activeTabIndex = this.tabs.indexOf(selectedTab);
        this.setComponent();
      }
    };

    this.handleTabKeyDown = event => {
      if (this.isHorizontal()) {
        switch (event.key) {
          case keyArrowLeft:
            event.preventDefault();
            this.adjustBackward(event);
            break;

          case keyArrowRight:
            event.preventDefault();
            this.adjustForward(event);
            break;
        }
      } else {
        switch (event.key) {
          case keyArrowUp:
            event.preventDefault();
            this.adjustBackward(event);
            break;

          case keyArrowDown:
            event.preventDefault();
            this.adjustForward(event);
            break;
        }
      }

      switch (event.key) {
        case keyHome:
          event.preventDefault();
          this.adjust(-this.activeTabIndex);
          break;

        case keyEnd:
          event.preventDefault();
          this.adjust(this.tabs.length - this.activeTabIndex - 1);
          break;
      }
    };

    this.adjustForward = e => {
      const group = this.tabs;
      let index = 0;
      index = this.activetab ? group.indexOf(this.activetab) + 1 : 1;

      if (index === group.length) {
        index = 0;
      }

      while (index < group.length && group.length > 1) {
        if (this.isFocusableElement(group[index])) {
          this.moveToTabByIndex(group, index);
          break;
        } else if (this.activetab && index === group.indexOf(this.activetab)) {
          break;
        } else if (index + 1 >= group.length) {
          index = 0;
        } else {
          index += 1;
        }
      }
    };

    this.adjustBackward = e => {
      const group = this.tabs;
      let index = 0;
      index = this.activetab ? group.indexOf(this.activetab) - 1 : 0;
      index = index < 0 ? group.length - 1 : index;

      while (index >= 0 && group.length > 1) {
        if (this.isFocusableElement(group[index])) {
          this.moveToTabByIndex(group, index);
          break;
        } else if (index - 1 < 0) {
          index = group.length - 1;
        } else {
          index -= 1;
        }
      }
    };

    this.moveToTabByIndex = (group, index) => {
      const tab = group[index];
      this.activetab = tab;
      this.prevActiveTabIndex = this.activeTabIndex;
      this.activeTabIndex = index;
      tab.focus();
      this.setComponent();
    };
  }
  /**
   * @internal
   */


  orientationChanged() {
    if (this.$fastController.isConnected) {
      this.setTabs();
      this.setTabPanels();
      this.handleActiveIndicatorPosition();
    }
  }
  /**
   * @internal
   */


  activeidChanged(oldValue, newValue) {
    if (this.$fastController.isConnected && this.tabs.length <= this.tabpanels.length) {
      this.prevActiveTabIndex = this.tabs.findIndex(item => item.id === oldValue);
      this.setTabs();
      this.setTabPanels();
      this.handleActiveIndicatorPosition();
    }
  }
  /**
   * @internal
   */


  tabsChanged() {
    if (this.$fastController.isConnected && this.tabs.length <= this.tabpanels.length) {
      this.tabIds = this.getTabIds();
      this.tabpanelIds = this.getTabPanelIds();
      this.setTabs();
      this.setTabPanels();
      this.handleActiveIndicatorPosition();
    }
  }
  /**
   * @internal
   */


  tabpanelsChanged() {
    if (this.$fastController.isConnected && this.tabpanels.length <= this.tabs.length) {
      this.tabIds = this.getTabIds();
      this.tabpanelIds = this.getTabPanelIds();
      this.setTabs();
      this.setTabPanels();
      this.handleActiveIndicatorPosition();
    }
  }

  getActiveIndex() {
    const id = this.activeid;

    if (id !== undefined) {
      return this.tabIds.indexOf(this.activeid) === -1 ? 0 : this.tabIds.indexOf(this.activeid);
    } else {
      return 0;
    }
  }

  getTabIds() {
    return this.tabs.map(tab => {
      var _a;

      return (_a = tab.getAttribute("id")) !== null && _a !== void 0 ? _a : `tab-${uniqueId()}`;
    });
  }

  getTabPanelIds() {
    return this.tabpanels.map(tabPanel => {
      var _a;

      return (_a = tabPanel.getAttribute("id")) !== null && _a !== void 0 ? _a : `panel-${uniqueId()}`;
    });
  }

  setComponent() {
    if (this.activeTabIndex !== this.prevActiveTabIndex) {
      this.activeid = this.tabIds[this.activeTabIndex];
      this.focusTab();
      this.change();
    }
  }

  isHorizontal() {
    return this.orientation === TabsOrientation.horizontal;
  }

  handleActiveIndicatorPosition() {
    // Ignore if we click twice on the same tab
    if (this.showActiveIndicator && this.activeindicator && this.activeTabIndex !== this.prevActiveTabIndex) {
      if (this.ticking) {
        this.ticking = false;
      } else {
        this.ticking = true;
        this.animateActiveIndicator();
      }
    }
  }

  animateActiveIndicator() {
    this.ticking = true;
    const gridProperty = this.isHorizontal() ? "gridColumn" : "gridRow";
    const translateProperty = this.isHorizontal() ? "translateX" : "translateY";
    const offsetProperty = this.isHorizontal() ? "offsetLeft" : "offsetTop";
    const prev = this.activeIndicatorRef[offsetProperty];
    this.activeIndicatorRef.style[gridProperty] = `${this.activeTabIndex + 1}`;
    const next = this.activeIndicatorRef[offsetProperty];
    this.activeIndicatorRef.style[gridProperty] = `${this.prevActiveTabIndex + 1}`;
    const dif = next - prev;
    this.activeIndicatorRef.style.transform = `${translateProperty}(${dif}px)`;
    this.activeIndicatorRef.classList.add("activeIndicatorTransition");
    this.activeIndicatorRef.addEventListener("transitionend", () => {
      this.ticking = false;
      this.activeIndicatorRef.style[gridProperty] = `${this.activeTabIndex + 1}`;
      this.activeIndicatorRef.style.transform = `${translateProperty}(0px)`;
      this.activeIndicatorRef.classList.remove("activeIndicatorTransition");
    });
  }
  /**
   * The adjust method for FASTTabs
   * @public
   * @remarks
   * This method allows the active index to be adjusted by numerical increments
   */


  adjust(adjustment) {
    this.prevActiveTabIndex = this.activeTabIndex;
    this.activeTabIndex = wrapInBounds(0, this.tabs.length - 1, this.activeTabIndex + adjustment);
    this.setComponent();
  }

  focusTab() {
    this.tabs[this.activeTabIndex].focus();
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.tabIds = this.getTabIds();
    this.tabpanelIds = this.getTabPanelIds();
    this.activeTabIndex = this.getActiveIndex();
  }

}

__decorate$1([attr], Tabs.prototype, "orientation", void 0);

__decorate$1([attr], Tabs.prototype, "activeid", void 0);

__decorate$1([observable], Tabs.prototype, "tabs", void 0);

__decorate$1([observable], Tabs.prototype, "tabpanels", void 0);

__decorate$1([attr({
  mode: "boolean"
})], Tabs.prototype, "activeindicator", void 0);

__decorate$1([observable], Tabs.prototype, "activeIndicatorRef", void 0);

__decorate$1([observable], Tabs.prototype, "showActiveIndicator", void 0);

applyMixins(Tabs, StartEnd);

class _TextArea extends FoundationElement {}
/**
 * A form-associated base class for the {@link @microsoft/fast-foundation#(TextArea:class)} component.
 *
 * @internal
 */


class FormAssociatedTextArea extends FormAssociated(_TextArea) {
  constructor() {
    super(...arguments);
    this.proxy = document.createElement("textarea");
  }

}

/**
 * Resize mode for a TextArea
 * @public
 */
const TextAreaResize = {
  /**
   * No resize.
   */
  none: "none",

  /**
   * Resize vertically and horizontally.
   */
  both: "both",

  /**
   * Resize horizontally.
   */
  horizontal: "horizontal",

  /**
   * Resize vertically.
   */
  vertical: "vertical"
};

/**
 * A Text Area Custom HTML Element.
 * Based largely on the {@link https://developer.mozilla.org/en-US/docs/Web/HTML/Element/textarea | <textarea> element }.
 *
 * @slot - The default slot for the label
 * @csspart label - The label
 * @csspart root - The element wrapping the control
 * @csspart control - The textarea element
 * @fires change - Emits a custom 'change' event when the textarea emits a change event
 *
 * @public
 */

class TextArea$1 extends FormAssociatedTextArea {
  constructor() {
    super(...arguments);
    /**
     * The resize mode of the element.
     * @public
     * @remarks
     * HTML Attribute: resize
     */

    this.resize = TextAreaResize.none;
    /**
     * Sizes the element horizontally by a number of character columns.
     *
     * @public
     * @remarks
     * HTML Attribute: cols
     */

    this.cols = 20;
    /**
     * @internal
     */

    this.handleTextInput = () => {
      this.value = this.control.value;
    };
  }

  readOnlyChanged() {
    if (this.proxy instanceof HTMLTextAreaElement) {
      this.proxy.readOnly = this.readOnly;
    }
  }

  autofocusChanged() {
    if (this.proxy instanceof HTMLTextAreaElement) {
      this.proxy.autofocus = this.autofocus;
    }
  }

  listChanged() {
    if (this.proxy instanceof HTMLTextAreaElement) {
      this.proxy.setAttribute("list", this.list);
    }
  }

  maxlengthChanged() {
    if (this.proxy instanceof HTMLTextAreaElement) {
      this.proxy.maxLength = this.maxlength;
    }
  }

  minlengthChanged() {
    if (this.proxy instanceof HTMLTextAreaElement) {
      this.proxy.minLength = this.minlength;
    }
  }

  spellcheckChanged() {
    if (this.proxy instanceof HTMLTextAreaElement) {
      this.proxy.spellcheck = this.spellcheck;
    }
  }
  /**
   * Selects all the text in the text area
   *
   * @public
   */


  select() {
    this.control.select();
    /**
     * The select event does not permeate the shadow DOM boundary.
     * This fn effectively proxies the select event,
     * emitting a `select` event whenever the internal
     * control emits a `select` event
     */

    this.$emit("select");
  }
  /**
   * Change event handler for inner control.
   * @remarks
   * "Change" events are not `composable` so they will not
   * permeate the shadow DOM boundary. This fn effectively proxies
   * the change event, emitting a `change` event whenever the internal
   * control emits a `change` event
   * @internal
   */


  handleChange() {
    this.$emit("change");
  }

}

__decorate$1([attr({
  mode: "boolean"
})], TextArea$1.prototype, "readOnly", void 0);

__decorate$1([attr], TextArea$1.prototype, "resize", void 0);

__decorate$1([attr({
  mode: "boolean"
})], TextArea$1.prototype, "autofocus", void 0);

__decorate$1([attr({
  attribute: "form"
})], TextArea$1.prototype, "formId", void 0);

__decorate$1([attr], TextArea$1.prototype, "list", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], TextArea$1.prototype, "maxlength", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter
})], TextArea$1.prototype, "minlength", void 0);

__decorate$1([attr], TextArea$1.prototype, "name", void 0);

__decorate$1([attr], TextArea$1.prototype, "placeholder", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter,
  mode: "fromView"
})], TextArea$1.prototype, "cols", void 0);

__decorate$1([attr({
  converter: nullableNumberConverter,
  mode: "fromView"
})], TextArea$1.prototype, "rows", void 0);

__decorate$1([attr({
  mode: "boolean"
})], TextArea$1.prototype, "spellcheck", void 0);

__decorate$1([observable], TextArea$1.prototype, "defaultSlottedNodes", void 0);

applyMixins(TextArea$1, DelegatesARIATextbox);

/**
 * The template for the {@link @microsoft/fast-foundation#(TextArea:class)} component.
 * @public
 */

const textAreaTemplate = (context, definition) => html`<template class=" ${x => x.readOnly ? "readonly" : ""} ${x => x.resize !== TextAreaResize.none ? `resize-${x.resize}` : ""}"><label part="label" for="control" class="${x => x.defaultSlottedNodes && x.defaultSlottedNodes.length ? "label" : "label label__hidden"}"><slot ${slotted("defaultSlottedNodes")}></slot></label><textarea part="control" class="control" id="control" ?autofocus="${x => x.autofocus}" cols="${x => x.cols}" ?disabled="${x => x.disabled}" form="${x => x.form}" list="${x => x.list}" maxlength="${x => x.maxlength}" minlength="${x => x.minlength}" name="${x => x.name}" placeholder="${x => x.placeholder}" ?readonly="${x => x.readOnly}" ?required="${x => x.required}" rows="${x => x.rows}" ?spellcheck="${x => x.spellcheck}" :value="${x => x.value}" aria-atomic="${x => x.ariaAtomic}" aria-busy="${x => x.ariaBusy}" aria-controls="${x => x.ariaControls}" aria-current="${x => x.ariaCurrent}" aria-describedby="${x => x.ariaDescribedby}" aria-details="${x => x.ariaDetails}" aria-disabled="${x => x.ariaDisabled}" aria-errormessage="${x => x.ariaErrormessage}" aria-flowto="${x => x.ariaFlowto}" aria-haspopup="${x => x.ariaHaspopup}" aria-hidden="${x => x.ariaHidden}" aria-invalid="${x => x.ariaInvalid}" aria-keyshortcuts="${x => x.ariaKeyshortcuts}" aria-label="${x => x.ariaLabel}" aria-labelledby="${x => x.ariaLabelledby}" aria-live="${x => x.ariaLive}" aria-owns="${x => x.ariaOwns}" aria-relevant="${x => x.ariaRelevant}" aria-roledescription="${x => x.ariaRoledescription}" @input="${(x, c) => x.handleTextInput()}" @change="${x => x.handleChange()}" ${ref("control")}></textarea></template>`;

/**
 * The template for the {@link @microsoft/fast-foundation#(TextField:class)} component.
 * @public
 */

const textFieldTemplate = (context, definition) => html`<template class=" ${x => x.readOnly ? "readonly" : ""} "><label part="label" for="control" class="${x => x.defaultSlottedNodes && x.defaultSlottedNodes.length ? "label" : "label label__hidden"}"><slot ${slotted({
  property: "defaultSlottedNodes",
  filter: whitespaceFilter
})}></slot></label><div class="root" part="root">${startSlotTemplate(context, definition)}<input class="control" part="control" id="control" @input="${x => x.handleTextInput()}" @change="${x => x.handleChange()}" ?autofocus="${x => x.autofocus}" ?disabled="${x => x.disabled}" list="${x => x.list}" maxlength="${x => x.maxlength}" minlength="${x => x.minlength}" pattern="${x => x.pattern}" placeholder="${x => x.placeholder}" ?readonly="${x => x.readOnly}" ?required="${x => x.required}" size="${x => x.size}" ?spellcheck="${x => x.spellcheck}" :value="${x => x.value}" type="${x => x.type}" aria-atomic="${x => x.ariaAtomic}" aria-busy="${x => x.ariaBusy}" aria-controls="${x => x.ariaControls}" aria-current="${x => x.ariaCurrent}" aria-describedby="${x => x.ariaDescribedby}" aria-details="${x => x.ariaDetails}" aria-disabled="${x => x.ariaDisabled}" aria-errormessage="${x => x.ariaErrormessage}" aria-flowto="${x => x.ariaFlowto}" aria-haspopup="${x => x.ariaHaspopup}" aria-hidden="${x => x.ariaHidden}" aria-invalid="${x => x.ariaInvalid}" aria-keyshortcuts="${x => x.ariaKeyshortcuts}" aria-label="${x => x.ariaLabel}" aria-labelledby="${x => x.ariaLabelledby}" aria-live="${x => x.ariaLive}" aria-owns="${x => x.ariaOwns}" aria-relevant="${x => x.ariaRelevant}" aria-roledescription="${x => x.ariaRoledescription}" ${ref("control")} />${endSlotTemplate(context, definition)}</div></template>`;

/**
 * The template for the {@link @microsoft/fast-foundation#(Toolbar:class)} component.
 *
 * @public
 */

const toolbarTemplate = (context, definition) => html`<template aria-label="${x => x.ariaLabel}" aria-labelledby="${x => x.ariaLabelledby}" aria-orientation="${x => x.orientation}" orientation="${x => x.orientation}" role="toolbar" @click="${(x, c) => x.clickHandler(c.event)}" @focusin="${(x, c) => x.focusinHandler(c.event)}" @keydown="${(x, c) => x.keydownHandler(c.event)}"><slot name="label"></slot><div class="positioning-region" part="positioning-region">${startSlotTemplate(context, definition)}<slot ${slotted({
  filter: elements(),
  property: "slottedItems"
})}></slot>${endSlotTemplate(context, definition)}</div></template>`;

/**
 * A map for directionality derived from keyboard input strings,
 * visual orientation, and text direction.
 *
 * @internal
 */

const ToolbarArrowKeyMap = Object.freeze({
  [ArrowKeys.ArrowUp]: {
    [Orientation.vertical]: -1
  },
  [ArrowKeys.ArrowDown]: {
    [Orientation.vertical]: 1
  },
  [ArrowKeys.ArrowLeft]: {
    [Orientation.horizontal]: {
      [Direction.ltr]: -1,
      [Direction.rtl]: 1
    }
  },
  [ArrowKeys.ArrowRight]: {
    [Orientation.horizontal]: {
      [Direction.ltr]: 1,
      [Direction.rtl]: -1
    }
  }
});
/**
 * A Toolbar Custom HTML Element.
 * Implements the {@link https://w3c.github.io/aria-practices/#Toolbar|ARIA Toolbar}.
 *
 * @slot start - Content which can be provided before the slotted items
 * @slot end - Content which can be provided after the slotted items
 * @slot - The default slot for slotted items
 * @slot label - The toolbar label
 * @csspart positioning-region - The element containing the items, start and end slots
 *
 * @public
 */

class Toolbar$1 extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * The internal index of the currently focused element.
     *
     * @internal
     */

    this._activeIndex = 0;
    /**
     * The text direction of the toolbar.
     *
     * @internal
     */

    this.direction = Direction.ltr;
    /**
     * The orientation of the toolbar.
     *
     * @public
     * @remarks
     * HTML Attribute: `orientation`
     */

    this.orientation = Orientation.horizontal;

    this.startEndSlotChange = () => {
      if (this.$fastController.isConnected) {
        this.reduceFocusableElements();
      }
    };
  }
  /**
   * The index of the currently focused element, clamped between 0 and the last element.
   *
   * @internal
   */


  get activeIndex() {
    Observable.track(this, "activeIndex");
    return this._activeIndex;
  }

  set activeIndex(value) {
    if (this.$fastController.isConnected) {
      this._activeIndex = limit(0, this.focusableElements.length - 1, value);
      Observable.notify(this, "activeIndex");
    }
  }

  slottedItemsChanged() {
    if (this.$fastController.isConnected) {
      this.reduceFocusableElements();
    }
  }
  /**
   * Set the activeIndex when a focusable element in the toolbar is clicked.
   *
   * @internal
   */


  clickHandler(e) {
    var _a;

    const activeIndex = (_a = this.focusableElements) === null || _a === void 0 ? void 0 : _a.indexOf(e.target);

    if (activeIndex > -1 && this.activeIndex !== activeIndex) {
      this.setFocusedElement(activeIndex);
    }

    return true;
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();
    this.direction = getDirection(this);
    this.start.addEventListener("slotchange", this.startEndSlotChange);
    this.end.addEventListener("slotchange", this.startEndSlotChange);
  }
  /**
   * @internal
   */


  disconnectedCallback() {
    super.disconnectedCallback();
    this.start.removeEventListener("slotchange", this.startEndSlotChange);
    this.end.removeEventListener("slotchange", this.startEndSlotChange);
  }
  /**
   * When the toolbar receives focus, set the currently active element as focused.
   *
   * @internal
   */


  focusinHandler(e) {
    const relatedTarget = e.relatedTarget;

    if (!relatedTarget || this.contains(relatedTarget)) {
      return;
    }

    this.setFocusedElement();
  }
  /**
   * Determines a value that can be used to iterate a list with the arrow keys.
   *
   * @param this - An element with an orientation and direction
   * @param key - The event key value
   * @internal
   */


  getDirectionalIncrementer(key) {
    var _a, _b, _c, _d, _e;

    return (_e = (_c = (_b = (_a = ToolbarArrowKeyMap[key]) === null || _a === void 0 ? void 0 : _a[this.orientation]) === null || _b === void 0 ? void 0 : _b[this.direction]) !== null && _c !== void 0 ? _c : (_d = ToolbarArrowKeyMap[key]) === null || _d === void 0 ? void 0 : _d[this.orientation]) !== null && _e !== void 0 ? _e : 0;
  }
  /**
   * Handle keyboard events for the toolbar.
   *
   * @internal
   */


  keydownHandler(e) {
    const key = e.key;

    if (!(key in ArrowKeys) || e.defaultPrevented || e.shiftKey) {
      return true;
    }

    const incrementer = this.getDirectionalIncrementer(key);

    if (!incrementer) {
      return !e.target.closest("[role=radiogroup]");
    }

    const nextIndex = this.activeIndex + incrementer;

    if (this.focusableElements[nextIndex]) {
      e.preventDefault();
    }

    this.setFocusedElement(nextIndex);
    return true;
  }
  /**
   * get all the slotted elements
   * @internal
   */


  get allSlottedItems() {
    return [...this.start.assignedElements(), ...this.slottedItems, ...this.end.assignedElements()];
  }
  /**
   * Prepare the slotted elements which can be focusable.
   *
   * @internal
   */


  reduceFocusableElements() {
    this.focusableElements = this.allSlottedItems.reduce(Toolbar$1.reduceFocusableItems, []);
    this.setFocusableElements();
  }
  /**
   * Set the activeIndex and focus the corresponding control.
   *
   * @param activeIndex - The new index to set
   * @internal
   */


  setFocusedElement(activeIndex = this.activeIndex) {
    var _a;

    this.activeIndex = activeIndex;
    this.setFocusableElements();
    (_a = this.focusableElements[this.activeIndex]) === null || _a === void 0 ? void 0 : _a.focus();
  }
  /**
   * Reduce a collection to only its focusable elements.
   *
   * @param elements - Collection of elements to reduce
   * @param element - The current element
   *
   * @internal
   */


  static reduceFocusableItems(elements, element) {
    var _a, _b, _c, _d;

    const isRoleRadio = element.getAttribute("role") === "radio";
    const isFocusableFastElement = (_b = (_a = element.$fastController) === null || _a === void 0 ? void 0 : _a.definition.shadowOptions) === null || _b === void 0 ? void 0 : _b.delegatesFocus;
    const hasFocusableShadow = Array.from((_d = (_c = element.shadowRoot) === null || _c === void 0 ? void 0 : _c.querySelectorAll("*")) !== null && _d !== void 0 ? _d : []).some(x => isFocusable(x));

    if (isFocusable(element) || isRoleRadio || isFocusableFastElement || hasFocusableShadow) {
      elements.push(element);
      return elements;
    }

    if (element.childElementCount) {
      return elements.concat(Array.from(element.children).reduce(Toolbar$1.reduceFocusableItems, []));
    }

    return elements;
  }
  /**
   * @internal
   */


  setFocusableElements() {
    if (this.$fastController.isConnected && this.focusableElements.length > 0) {
      this.focusableElements.forEach((element, index) => {
        element.tabIndex = this.activeIndex === index ? 0 : -1;
      });
    }
  }

}

__decorate$1([observable], Toolbar$1.prototype, "direction", void 0);

__decorate$1([attr], Toolbar$1.prototype, "orientation", void 0);

__decorate$1([observable], Toolbar$1.prototype, "slottedItems", void 0);

__decorate$1([observable], Toolbar$1.prototype, "slottedLabel", void 0);
/**
 * Includes ARIA states and properties relating to the ARIA toolbar role
 *
 * @public
 */


class DelegatesARIAToolbar {}

__decorate$1([attr({
  attribute: "aria-labelledby"
})], DelegatesARIAToolbar.prototype, "ariaLabelledby", void 0);

__decorate$1([attr({
  attribute: "aria-label"
})], DelegatesARIAToolbar.prototype, "ariaLabel", void 0);

applyMixins(DelegatesARIAToolbar, ARIAGlobalStatesAndProperties);
applyMixins(Toolbar$1, StartEnd, DelegatesARIAToolbar);

/**
 * Creates a template for the {@link @microsoft/fast-foundation#(Tooltip:class)} component using the provided prefix.
 * @public
 */

const tooltipTemplate = (context, definition) => {
  return html` ${when(x => x.tooltipVisible, html`<${context.tagFor(AnchoredRegion)} fixed-placement="true" auto-update-mode="${x => x.autoUpdateMode}" vertical-positioning-mode="${x => x.verticalPositioningMode}" vertical-default-position="${x => x.verticalDefaultPosition}" vertical-inset="${x => x.verticalInset}" vertical-scaling="${x => x.verticalScaling}" horizontal-positioning-mode="${x => x.horizontalPositioningMode}" horizontal-default-position="${x => x.horizontalDefaultPosition}" horizontal-scaling="${x => x.horizontalScaling}" horizontal-inset="${x => x.horizontalInset}" vertical-viewport-lock="${x => x.horizontalViewportLock}" horizontal-viewport-lock="${x => x.verticalViewportLock}" dir="${x => x.currentDirection}" ${ref("region")}><div class="tooltip" part="tooltip" role="tooltip"><slot></slot></div></${context.tagFor(AnchoredRegion)}>`)} `;
};

/**
 * Enumerates possible tooltip positions
 *
 * @public
 */
const TooltipPosition = {
  /**
   * The tooltip is positioned above the element
   */
  top: "top",

  /**
   * The tooltip is positioned to the right of the element
   */
  right: "right",

  /**
   * The tooltip is positioned below the element
   */
  bottom: "bottom",

  /**
   * The tooltip is positioned to the left of the element
   */
  left: "left",

  /**
   * The tooltip is positioned before the element
   */
  start: "start",

  /**
   * The tooltip is positioned after the element
   */
  end: "end",

  /**
   * The tooltip is positioned above the element and to the left
   */
  topLeft: "top-left",

  /**
   * The tooltip is positioned above the element and to the right
   */
  topRight: "top-right",

  /**
   * The tooltip is positioned below the element and to the left
   */
  bottomLeft: "bottom-left",

  /**
   * The tooltip is positioned below the element and to the right
   */
  bottomRight: "bottom-right",

  /**
   * The tooltip is positioned above the element and to the left
   */
  topStart: "top-start",

  /**
   * The tooltip is positioned above the element and to the right
   */
  topEnd: "top-end",

  /**
   * The tooltip is positioned below the element and to the left
   */
  bottomStart: "bottom-start",

  /**
   * The tooltip is positioned below the element and to the right
   */
  bottomEnd: "bottom-end"
};

/**
 * An Tooltip Custom HTML Element.
 *
 * @slot - The default slot for the tooltip content
 * @csspart tooltip - The tooltip element
 * @fires dismiss - Fires a custom 'dismiss' event when the tooltip is visible and escape key is pressed
 *
 * @public
 */

class Tooltip extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * The id of the element the tooltip is anchored to
     *
     * @defaultValue - undefined
     * @public
     * HTML Attribute: anchor
     */

    this.anchor = "";
    /**
     * The delay in milliseconds before a tooltip is shown after a hover event
     *
     * @defaultValue - 300
     * @public
     * HTML Attribute: delay
     */

    this.delay = 300;
    /**
     * Controls when the tooltip updates its position, default is 'anchor' which only updates when
     * the anchor is resized.  'auto' will update on scroll/resize events.
     * Corresponds to anchored-region auto-update-mode.
     * @public
     * @remarks
     * HTML Attribute: auto-update-mode
     */

    this.autoUpdateMode = "anchor";
    /**
     * the html element currently being used as anchor.
     * Setting this directly overrides the anchor attribute.
     *
     * @public
     */

    this.anchorElement = null;
    /**
     * The current viewport element instance
     *
     * @internal
     */

    this.viewportElement = null;
    /**
     * @internal
     * @defaultValue "dynamic"
     */

    this.verticalPositioningMode = "dynamic";
    /**
     * @internal
     * @defaultValue "dynamic"
     */

    this.horizontalPositioningMode = "dynamic";
    /**
     * @internal
     */

    this.horizontalInset = "false";
    /**
     * @internal
     */

    this.verticalInset = "false";
    /**
     * @internal
     */

    this.horizontalScaling = "content";
    /**
     * @internal
     */

    this.verticalScaling = "content";
    /**
     * @internal
     */

    this.verticalDefaultPosition = undefined;
    /**
     * @internal
     */

    this.horizontalDefaultPosition = undefined;
    /**
     * @internal
     */

    this.tooltipVisible = false;
    /**
     * Track current direction to pass to the anchored region
     * updated when tooltip is shown
     *
     * @internal
     */

    this.currentDirection = Direction.ltr;
    /**
     * The timer that tracks delay time before the tooltip is shown on hover
     */

    this.showDelayTimer = null;
    /**
     * The timer that tracks delay time before the tooltip is hidden
     */

    this.hideDelayTimer = null;
    /**
     * Indicates whether the anchor is currently being hovered or has focus
     */

    this.isAnchorHoveredFocused = false;
    /**
     * Indicates whether the region is currently being hovered
     */

    this.isRegionHovered = false;
    /**
     * invoked when the anchored region's position relative to the anchor changes
     *
     * @internal
     */

    this.handlePositionChange = ev => {
      this.classList.toggle("top", this.region.verticalPosition === "start");
      this.classList.toggle("bottom", this.region.verticalPosition === "end");
      this.classList.toggle("inset-top", this.region.verticalPosition === "insetStart");
      this.classList.toggle("inset-bottom", this.region.verticalPosition === "insetEnd");
      this.classList.toggle("center-vertical", this.region.verticalPosition === "center");
      this.classList.toggle("left", this.region.horizontalPosition === "start");
      this.classList.toggle("right", this.region.horizontalPosition === "end");
      this.classList.toggle("inset-left", this.region.horizontalPosition === "insetStart");
      this.classList.toggle("inset-right", this.region.horizontalPosition === "insetEnd");
      this.classList.toggle("center-horizontal", this.region.horizontalPosition === "center");
    };
    /**
     * mouse enters region
     */


    this.handleRegionMouseOver = ev => {
      this.isRegionHovered = true;
    };
    /**
     * mouse leaves region
     */


    this.handleRegionMouseOut = ev => {
      this.isRegionHovered = false;
      this.startHideDelayTimer();
    };
    /**
     * mouse enters anchor
     */


    this.handleAnchorMouseOver = ev => {
      if (this.tooltipVisible) {
        // tooltip is already visible, just set the anchor hover flag
        this.isAnchorHoveredFocused = true;
        return;
      }

      this.startShowDelayTimer();
    };
    /**
     * mouse leaves anchor
     */


    this.handleAnchorMouseOut = ev => {
      this.isAnchorHoveredFocused = false;
      this.clearShowDelayTimer();
      this.startHideDelayTimer();
    };
    /**
     * anchor gets focus
     */


    this.handleAnchorFocusIn = ev => {
      this.startShowDelayTimer();
    };
    /**
     * anchor loses focus
     */


    this.handleAnchorFocusOut = ev => {
      this.isAnchorHoveredFocused = false;
      this.clearShowDelayTimer();
      this.startHideDelayTimer();
    };
    /**
     * starts the hide timer
     */


    this.startHideDelayTimer = () => {
      this.clearHideDelayTimer();

      if (!this.tooltipVisible) {
        return;
      } // allow 60 ms for account for pointer to move between anchor/tooltip
      // without hiding tooltip


      this.hideDelayTimer = window.setTimeout(() => {
        this.updateTooltipVisibility();
      }, 60);
    };
    /**
     * clears the hide delay
     */


    this.clearHideDelayTimer = () => {
      if (this.hideDelayTimer !== null) {
        clearTimeout(this.hideDelayTimer);
        this.hideDelayTimer = null;
      }
    };
    /**
     * starts the show timer if not currently running
     */


    this.startShowDelayTimer = () => {
      if (this.isAnchorHoveredFocused) {
        return;
      }

      if (this.delay > 1) {
        if (this.showDelayTimer === null) this.showDelayTimer = window.setTimeout(() => {
          this.startHover();
        }, this.delay);
        return;
      }

      this.startHover();
    };
    /**
     * start hover
     */


    this.startHover = () => {
      this.isAnchorHoveredFocused = true;
      this.updateTooltipVisibility();
    };
    /**
     * clears the show delay
     */


    this.clearShowDelayTimer = () => {
      if (this.showDelayTimer !== null) {
        clearTimeout(this.showDelayTimer);
        this.showDelayTimer = null;
      }
    };
    /**
     *  Gets the anchor element by id
     */


    this.getAnchor = () => {
      const rootNode = this.getRootNode();

      if (rootNode instanceof ShadowRoot) {
        return rootNode.getElementById(this.anchor);
      }

      return document.getElementById(this.anchor);
    };
    /**
     * handles key down events to check for dismiss
     */


    this.handleDocumentKeydown = e => {
      if (!e.defaultPrevented && this.tooltipVisible) {
        switch (e.key) {
          case keyEscape:
            this.isAnchorHoveredFocused = false;
            this.updateTooltipVisibility();
            this.$emit("dismiss");
            break;
        }
      }
    };
    /**
     * determines whether to show or hide the tooltip based on current state
     */


    this.updateTooltipVisibility = () => {
      if (this.visible === false) {
        this.hideTooltip();
      } else if (this.visible === true) {
        this.showTooltip();
        return;
      } else {
        if (this.isAnchorHoveredFocused || this.isRegionHovered) {
          this.showTooltip();
          return;
        }

        this.hideTooltip();
      }
    };
    /**
     * shows the tooltip
     */


    this.showTooltip = () => {
      if (this.tooltipVisible) {
        return;
      }

      this.currentDirection = getDirection(this);
      this.tooltipVisible = true;
      document.addEventListener("keydown", this.handleDocumentKeydown);
      DOM.queueUpdate(this.setRegionProps);
    };
    /**
     * hides the tooltip
     */


    this.hideTooltip = () => {
      if (!this.tooltipVisible) {
        return;
      }

      this.clearHideDelayTimer();

      if (this.region !== null && this.region !== undefined) {
        this.region.removeEventListener("positionchange", this.handlePositionChange);
        this.region.viewportElement = null;
        this.region.anchorElement = null;
        this.region.removeEventListener("mouseover", this.handleRegionMouseOver);
        this.region.removeEventListener("mouseout", this.handleRegionMouseOut);
      }

      document.removeEventListener("keydown", this.handleDocumentKeydown);
      this.tooltipVisible = false;
    };
    /**
     * updates the tooltip anchored region props after it has been
     * added to the DOM
     */


    this.setRegionProps = () => {
      if (!this.tooltipVisible) {
        return;
      }

      this.region.viewportElement = this.viewportElement;
      this.region.anchorElement = this.anchorElement;
      this.region.addEventListener("positionchange", this.handlePositionChange);
      this.region.addEventListener("mouseover", this.handleRegionMouseOver, {
        passive: true
      });
      this.region.addEventListener("mouseout", this.handleRegionMouseOut, {
        passive: true
      });
    };
  }

  visibleChanged() {
    if (this.$fastController.isConnected) {
      this.updateTooltipVisibility();
      this.updateLayout();
    }
  }

  anchorChanged() {
    if (this.$fastController.isConnected) {
      this.anchorElement = this.getAnchor();
    }
  }

  positionChanged() {
    if (this.$fastController.isConnected) {
      this.updateLayout();
    }
  }

  anchorElementChanged(oldValue) {
    if (this.$fastController.isConnected) {
      if (oldValue !== null && oldValue !== undefined) {
        oldValue.removeEventListener("mouseover", this.handleAnchorMouseOver);
        oldValue.removeEventListener("mouseout", this.handleAnchorMouseOut);
        oldValue.removeEventListener("focusin", this.handleAnchorFocusIn);
        oldValue.removeEventListener("focusout", this.handleAnchorFocusOut);
      }

      if (this.anchorElement !== null && this.anchorElement !== undefined) {
        this.anchorElement.addEventListener("mouseover", this.handleAnchorMouseOver, {
          passive: true
        });
        this.anchorElement.addEventListener("mouseout", this.handleAnchorMouseOut, {
          passive: true
        });
        this.anchorElement.addEventListener("focusin", this.handleAnchorFocusIn, {
          passive: true
        });
        this.anchorElement.addEventListener("focusout", this.handleAnchorFocusOut, {
          passive: true
        });
        const anchorId = this.anchorElement.id;

        if (this.anchorElement.parentElement !== null) {
          this.anchorElement.parentElement.querySelectorAll(":hover").forEach(element => {
            if (element.id === anchorId) {
              this.startShowDelayTimer();
            }
          });
        }
      }

      if (this.region !== null && this.region !== undefined && this.tooltipVisible) {
        this.region.anchorElement = this.anchorElement;
      }

      this.updateLayout();
    }
  }

  viewportElementChanged() {
    if (this.region !== null && this.region !== undefined) {
      this.region.viewportElement = this.viewportElement;
    }

    this.updateLayout();
  }

  connectedCallback() {
    super.connectedCallback();
    this.anchorElement = this.getAnchor();
    this.updateTooltipVisibility();
  }

  disconnectedCallback() {
    this.hideTooltip();
    this.clearShowDelayTimer();
    this.clearHideDelayTimer();
    super.disconnectedCallback();
  }
  /**
   * updated the properties being passed to the anchored region
   */


  updateLayout() {
    this.verticalPositioningMode = "locktodefault";
    this.horizontalPositioningMode = "locktodefault";

    switch (this.position) {
      case TooltipPosition.top:
      case TooltipPosition.bottom:
        this.verticalDefaultPosition = this.position;
        this.horizontalDefaultPosition = "center";
        break;

      case TooltipPosition.right:
      case TooltipPosition.left:
      case TooltipPosition.start:
      case TooltipPosition.end:
        this.verticalDefaultPosition = "center";
        this.horizontalDefaultPosition = this.position;
        break;

      case TooltipPosition.topLeft:
        this.verticalDefaultPosition = "top";
        this.horizontalDefaultPosition = "left";
        break;

      case TooltipPosition.topRight:
        this.verticalDefaultPosition = "top";
        this.horizontalDefaultPosition = "right";
        break;

      case TooltipPosition.bottomLeft:
        this.verticalDefaultPosition = "bottom";
        this.horizontalDefaultPosition = "left";
        break;

      case TooltipPosition.bottomRight:
        this.verticalDefaultPosition = "bottom";
        this.horizontalDefaultPosition = "right";
        break;

      case TooltipPosition.topStart:
        this.verticalDefaultPosition = "top";
        this.horizontalDefaultPosition = "start";
        break;

      case TooltipPosition.topEnd:
        this.verticalDefaultPosition = "top";
        this.horizontalDefaultPosition = "end";
        break;

      case TooltipPosition.bottomStart:
        this.verticalDefaultPosition = "bottom";
        this.horizontalDefaultPosition = "start";
        break;

      case TooltipPosition.bottomEnd:
        this.verticalDefaultPosition = "bottom";
        this.horizontalDefaultPosition = "end";
        break;

      default:
        this.verticalPositioningMode = "dynamic";
        this.horizontalPositioningMode = "dynamic";
        this.verticalDefaultPosition = void 0;
        this.horizontalDefaultPosition = "center";
        break;
    }
  }

}

__decorate$1([attr({
  mode: "boolean"
})], Tooltip.prototype, "visible", void 0);

__decorate$1([attr], Tooltip.prototype, "anchor", void 0);

__decorate$1([attr], Tooltip.prototype, "delay", void 0);

__decorate$1([attr], Tooltip.prototype, "position", void 0);

__decorate$1([attr({
  attribute: "auto-update-mode"
})], Tooltip.prototype, "autoUpdateMode", void 0);

__decorate$1([attr({
  attribute: "horizontal-viewport-lock"
})], Tooltip.prototype, "horizontalViewportLock", void 0);

__decorate$1([attr({
  attribute: "vertical-viewport-lock"
})], Tooltip.prototype, "verticalViewportLock", void 0);

__decorate$1([observable], Tooltip.prototype, "anchorElement", void 0);

__decorate$1([observable], Tooltip.prototype, "viewportElement", void 0);

__decorate$1([observable], Tooltip.prototype, "verticalPositioningMode", void 0);

__decorate$1([observable], Tooltip.prototype, "horizontalPositioningMode", void 0);

__decorate$1([observable], Tooltip.prototype, "horizontalInset", void 0);

__decorate$1([observable], Tooltip.prototype, "verticalInset", void 0);

__decorate$1([observable], Tooltip.prototype, "horizontalScaling", void 0);

__decorate$1([observable], Tooltip.prototype, "verticalScaling", void 0);

__decorate$1([observable], Tooltip.prototype, "verticalDefaultPosition", void 0);

__decorate$1([observable], Tooltip.prototype, "horizontalDefaultPosition", void 0);

__decorate$1([observable], Tooltip.prototype, "tooltipVisible", void 0);

__decorate$1([observable], Tooltip.prototype, "currentDirection", void 0);

/**
 * The template for the {@link @microsoft/fast-foundation#(TreeItem:class)} component.
 * @public
 */

const treeItemTemplate = (context, definition) => html`<template role="treeitem" slot="${x => x.isNestedItem() ? "item" : void 0}" tabindex="-1" class="${x => x.expanded ? "expanded" : ""} ${x => x.selected ? "selected" : ""} ${x => x.nested ? "nested" : ""} ${x => x.disabled ? "disabled" : ""}" aria-expanded="${x => x.childItems && x.childItemLength() > 0 ? x.expanded : void 0}" aria-selected="${x => x.selected}" aria-disabled="${x => x.disabled}" @focusin="${(x, c) => x.handleFocus(c.event)}" @focusout="${(x, c) => x.handleBlur(c.event)}" ${children({
  property: "childItems",
  filter: elements()
})}><div class="positioning-region" part="positioning-region"><div class="content-region" part="content-region">${when(x => x.childItems && x.childItemLength() > 0, html`<div aria-hidden="true" class="expand-collapse-button" part="expand-collapse-button" @click="${(x, c) => x.handleExpandCollapseButtonClick(c.event)}" ${ref("expandCollapseButton")}><slot name="expand-collapse-glyph">${definition.expandCollapseGlyph || ""}</slot></div>`)} ${startSlotTemplate(context, definition)}<slot></slot>${endSlotTemplate(context, definition)}</div></div>${when(x => x.childItems && x.childItemLength() > 0 && (x.expanded || x.renderCollapsedChildren), html`<div role="group" class="items" part="items"><slot name="item" ${slotted("items")}></slot></div>`)}</template>`;

/**
 * check if the item is a tree item
 * @public
 * @remarks
 * determines if element is an HTMLElement and if it has the role treeitem
 */

function isTreeItemElement(el) {
  return isHTMLElement(el) && el.getAttribute("role") === "treeitem";
}
/**
 * A Tree item Custom HTML Element.
 *
 * @slot start - Content which can be provided before the tree item content
 * @slot end - Content which can be provided after the tree item content
 * @slot - The default slot for tree item text content
 * @slot item - The slot for tree items (fast tree items manage this assignment themselves)
 * @slot expand-collapse-button - The expand/collapse button
 * @csspart positioning-region - The element used to position the tree item content with exception of any child nodes
 * @csspart content-region - The element containing the expand/collapse, start, and end slots
 * @csspart items - The element wrapping any child items
 * @csspart expand-collapse-button - The expand/collapse button
 * @fires expanded-change - Fires a custom 'expanded-change' event when the expanded state changes
 * @fires selected-change - Fires a custom 'selected-change' event when the selected state changes
 *
 * @public
 */

class TreeItem extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * When true, the control will be appear expanded by user interaction.
     * @public
     * @remarks
     * HTML Attribute: expanded
     */

    this.expanded = false;
    /**
     * Whether the item is focusable
     *
     * @internal
     */

    this.focusable = false;
    /**
     * Whether the tree is nested
     *
     * @public
     */

    this.isNestedItem = () => {
      return isTreeItemElement(this.parentElement);
    };
    /**
     * Handle expand button click
     *
     * @internal
     */


    this.handleExpandCollapseButtonClick = e => {
      if (!this.disabled && !e.defaultPrevented) {
        this.expanded = !this.expanded;
      }
    };
    /**
     * Handle focus events
     *
     * @internal
     */


    this.handleFocus = e => {
      this.setAttribute("tabindex", "0");
    };
    /**
     * Handle blur events
     *
     * @internal
     */


    this.handleBlur = e => {
      this.setAttribute("tabindex", "-1");
    };
  }

  expandedChanged() {
    if (this.$fastController.isConnected) {
      this.$emit("expanded-change", this);
    }
  }

  selectedChanged() {
    if (this.$fastController.isConnected) {
      this.$emit("selected-change", this);
    }
  }

  itemsChanged(oldValue, newValue) {
    if (this.$fastController.isConnected) {
      this.items.forEach(node => {
        if (isTreeItemElement(node)) {
          // TODO: maybe not require it to be a TreeItem?
          node.nested = true;
        }
      });
    }
  }
  /**
   * Places document focus on a tree item
   *
   * @public
   * @param el - the element to focus
   */


  static focusItem(el) {
    el.focusable = true;
    el.focus();
  }
  /**
   * Gets number of children
   *
   * @internal
   */


  childItemLength() {
    const treeChildren = this.childItems.filter(item => {
      return isTreeItemElement(item);
    });
    return treeChildren ? treeChildren.length : 0;
  }

}

__decorate$1([attr({
  mode: "boolean"
})], TreeItem.prototype, "expanded", void 0);

__decorate$1([attr({
  mode: "boolean"
})], TreeItem.prototype, "selected", void 0);

__decorate$1([attr({
  mode: "boolean"
})], TreeItem.prototype, "disabled", void 0);

__decorate$1([observable], TreeItem.prototype, "focusable", void 0);

__decorate$1([observable], TreeItem.prototype, "childItems", void 0);

__decorate$1([observable], TreeItem.prototype, "items", void 0);

__decorate$1([observable], TreeItem.prototype, "nested", void 0);

__decorate$1([observable], TreeItem.prototype, "renderCollapsedChildren", void 0);

applyMixins(TreeItem, StartEnd);

/**
 * The template for the {@link @microsoft/fast-foundation#TreeView} component.
 * @public
 */

const treeViewTemplate = (context, definition) => html`<template role="tree" ${ref("treeView")} @keydown="${(x, c) => x.handleKeyDown(c.event)}" @focusin="${(x, c) => x.handleFocus(c.event)}" @focusout="${(x, c) => x.handleBlur(c.event)}" @click="${(x, c) => x.handleClick(c.event)}" @selected-change="${(x, c) => x.handleSelectedChange(c.event)}"><slot ${slotted("slottedTreeItems")}></slot></template>`;

/**
 * A Tree view Custom HTML Element.
 * Implements the {@link https://w3c.github.io/aria-practices/#TreeView | ARIA TreeView }.
 *
 * @slot - The default slot for tree items
 *
 * @public
 */

class TreeView extends FoundationElement {
  constructor() {
    super(...arguments);
    /**
     * The tree item that is designated to be in the tab queue.
     *
     * @internal
     */

    this.currentFocused = null;
    /**
     * Handle focus events
     *
     * @internal
     */

    this.handleFocus = e => {
      if (this.slottedTreeItems.length < 1) {
        // no child items, nothing to do
        return;
      }

      if (e.target === this) {
        if (this.currentFocused === null) {
          this.currentFocused = this.getValidFocusableItem();
        }

        if (this.currentFocused !== null) {
          TreeItem.focusItem(this.currentFocused);
        }

        return;
      }

      if (this.contains(e.target)) {
        this.setAttribute("tabindex", "-1");
        this.currentFocused = e.target;
      }
    };
    /**
     * Handle blur events
     *
     * @internal
     */


    this.handleBlur = e => {
      if (e.target instanceof HTMLElement && (e.relatedTarget === null || !this.contains(e.relatedTarget))) {
        this.setAttribute("tabindex", "0");
      }
    };
    /**
     * KeyDown handler
     *
     *  @internal
     */


    this.handleKeyDown = e => {
      if (e.defaultPrevented) {
        return;
      }

      if (this.slottedTreeItems.length < 1) {
        return true;
      }

      const treeItems = this.getVisibleNodes();

      switch (e.key) {
        case keyHome:
          if (treeItems.length) {
            TreeItem.focusItem(treeItems[0]);
          }

          return;

        case keyEnd:
          if (treeItems.length) {
            TreeItem.focusItem(treeItems[treeItems.length - 1]);
          }

          return;

        case keyArrowLeft:
          if (e.target && this.isFocusableElement(e.target)) {
            const item = e.target;

            if (item instanceof TreeItem && item.childItemLength() > 0 && item.expanded) {
              item.expanded = false;
            } else if (item instanceof TreeItem && item.parentElement instanceof TreeItem) {
              TreeItem.focusItem(item.parentElement);
            }
          }

          return false;

        case keyArrowRight:
          if (e.target && this.isFocusableElement(e.target)) {
            const item = e.target;

            if (item instanceof TreeItem && item.childItemLength() > 0 && !item.expanded) {
              item.expanded = true;
            } else if (item instanceof TreeItem && item.childItemLength() > 0) {
              this.focusNextNode(1, e.target);
            }
          }

          return;

        case keyArrowDown:
          if (e.target && this.isFocusableElement(e.target)) {
            this.focusNextNode(1, e.target);
          }

          return;

        case keyArrowUp:
          if (e.target && this.isFocusableElement(e.target)) {
            this.focusNextNode(-1, e.target);
          }

          return;

        case keyEnter:
          // In single-select trees where selection does not follow focus (see note below),
          // the default action is typically to select the focused node.
          this.handleClick(e);
          return;
      } // don't prevent default if we took no action


      return true;
    };
    /**
     * Handles the selected-changed events bubbling up
     * from child tree items
     *
     *  @internal
     */


    this.handleSelectedChange = e => {
      if (e.defaultPrevented) {
        return;
      }

      if (!(e.target instanceof Element) || !isTreeItemElement(e.target)) {
        return true;
      }

      const item = e.target;

      if (item.selected) {
        if (this.currentSelected && this.currentSelected !== item) {
          this.currentSelected.selected = false;
        } // new selected item


        this.currentSelected = item;
      } else if (!item.selected && this.currentSelected === item) {
        // selected item deselected
        this.currentSelected = null;
      }

      return;
    };
    /**
     * Updates the tree view when slottedTreeItems changes
     */


    this.setItems = () => {
      // force single selection
      // defaults to first one found
      const selectedItem = this.treeView.querySelector("[aria-selected='true']");
      this.currentSelected = selectedItem; // invalidate the current focused item if it is no longer valid

      if (this.currentFocused === null || !this.contains(this.currentFocused)) {
        this.currentFocused = this.getValidFocusableItem();
      } // toggle properties on child elements


      this.nested = this.checkForNestedItems();
      const treeItems = this.getVisibleNodes();
      treeItems.forEach(node => {
        if (isTreeItemElement(node)) {
          node.nested = this.nested;
        }
      });
    };
    /**
     * check if the item is focusable
     */


    this.isFocusableElement = el => {
      return isTreeItemElement(el);
    };

    this.isSelectedElement = el => {
      return el.selected;
    };
  }

  slottedTreeItemsChanged() {
    if (this.$fastController.isConnected) {
      // update for slotted children change
      this.setItems();
    }
  }

  connectedCallback() {
    super.connectedCallback();
    this.setAttribute("tabindex", "0");
    DOM.queueUpdate(() => {
      this.setItems();
    });
  }
  /**
   * Handles click events bubbling up
   *
   *  @internal
   */


  handleClick(e) {
    if (e.defaultPrevented) {
      // handled, do nothing
      return;
    }

    if (!(e.target instanceof Element) || !isTreeItemElement(e.target)) {
      // not a tree item, ignore
      return true;
    }

    const item = e.target;

    if (!item.disabled) {
      item.selected = !item.selected;
    }

    return;
  }
  /**
   * Move focus to a tree item based on its offset from the provided item
   */


  focusNextNode(delta, item) {
    const visibleNodes = this.getVisibleNodes();

    if (!visibleNodes) {
      return;
    }

    const focusItem = visibleNodes[visibleNodes.indexOf(item) + delta];

    if (isHTMLElement(focusItem)) {
      TreeItem.focusItem(focusItem);
    }
  }
  /**
   * checks if there are any nested tree items
   */


  getValidFocusableItem() {
    const treeItems = this.getVisibleNodes(); // default to selected element if there is one

    let focusIndex = treeItems.findIndex(this.isSelectedElement);

    if (focusIndex === -1) {
      // otherwise first focusable tree item
      focusIndex = treeItems.findIndex(this.isFocusableElement);
    }

    if (focusIndex !== -1) {
      return treeItems[focusIndex];
    }

    return null;
  }
  /**
   * checks if there are any nested tree items
   */


  checkForNestedItems() {
    return this.slottedTreeItems.some(node => {
      return isTreeItemElement(node) && node.querySelector("[role='treeitem']");
    });
  }

  getVisibleNodes() {
    return getDisplayedNodes(this, "[role='treeitem']") || [];
  }

}

__decorate$1([attr({
  attribute: "render-collapsed-nodes"
})], TreeView.prototype, "renderCollapsedNodes", void 0);

__decorate$1([observable], TreeView.prototype, "currentSelected", void 0);

__decorate$1([observable], TreeView.prototype, "slottedTreeItems", void 0);

/**
 * An abstract behavior to react to media queries. Implementations should implement
 * the `constructListener` method to perform some action based on media query changes.
 *
 * @public
 */
class MatchMediaBehavior {
  /**
   *
   * @param query - The media query to operate from.
   */
  constructor(query) {
    /**
     * The behavior needs to operate on element instances but elements might share a behavior instance.
     * To ensure proper attachment / detachment per instance, we construct a listener for
     * each bind invocation and cache the listeners by element reference.
     */
    this.listenerCache = new WeakMap();
    this.query = query;
  }
  /**
   * Binds the behavior to the element.
   * @param source - The element for which the behavior is bound.
   */


  bind(source) {
    const {
      query
    } = this;
    const listener = this.constructListener(source); // Invoke immediately to add if the query currently matches

    listener.bind(query)();
    query.addListener(listener);
    this.listenerCache.set(source, listener);
  }
  /**
   * Unbinds the behavior from the element.
   * @param source - The element for which the behavior is unbinding.
   */


  unbind(source) {
    const listener = this.listenerCache.get(source);

    if (listener) {
      this.query.removeListener(listener);
      this.listenerCache.delete(source);
    }
  }

}
/**
 * A behavior to add or remove a stylesheet from an element based on a media query. The behavior ensures that
 * styles are applied while the a query matches the environment and that styles are not applied if the query does
 * not match the environment.
 *
 * @public
 */

class MatchMediaStyleSheetBehavior extends MatchMediaBehavior {
  /**
   * Constructs a {@link MatchMediaStyleSheetBehavior} instance.
   * @param query - The media query to operate from.
   * @param styles - The styles to coordinate with the query.
   */
  constructor(query, styles) {
    super(query);
    this.styles = styles;
  }
  /**
   * Defines a function to construct {@link MatchMediaStyleSheetBehavior | MatchMediaStyleSheetBehaviors} for
   * a provided query.
   * @param query - The media query to operate from.
   *
   * @public
   * @example
   *
   * ```ts
   * import { css } from "@microsoft/fast-element";
   * import { MatchMediaStyleSheetBehavior } from "@microsoft/fast-foundation";
   *
   * const landscapeBehavior = MatchMediaStyleSheetBehavior.with(
   *   window.matchMedia("(orientation: landscape)")
   * );
   * const styles = css`
   *   :host {
   *     width: 200px;
   *     height: 400px;
   *   }
   * `
   * .withBehaviors(landscapeBehavior(css`
   *   :host {
   *     width: 400px;
   *     height: 200px;
   *   }
   * `))
   * ```
   */


  static with(query) {
    return styles => {
      return new MatchMediaStyleSheetBehavior(query, styles);
    };
  }
  /**
   * Constructs a match-media listener for a provided element.
   * @param source - the element for which to attach or detach styles.
   * @internal
   */


  constructListener(source) {
    let attached = false;
    const styles = this.styles;
    return function listener() {
      const {
        matches
      } = this;

      if (matches && !attached) {
        source.$fastController.addStyles(styles);
        attached = matches;
      } else if (!matches && attached) {
        source.$fastController.removeStyles(styles);
        attached = matches;
      }
    };
  }
  /**
   * Unbinds the behavior from the element.
   * @param source - The element for which the behavior is unbinding.
   * @internal
   */


  unbind(source) {
    super.unbind(source);
    source.$fastController.removeStyles(this.styles);
  }

}
/**
 * This can be used to construct a behavior to apply a forced-colors only stylesheet.
 * @public
 */

const forcedColorsStylesheetBehavior = MatchMediaStyleSheetBehavior.with(window.matchMedia("(forced-colors)"));
/**
 * This can be used to construct a behavior to apply a prefers color scheme: dark only stylesheet.
 * @public
 */

MatchMediaStyleSheetBehavior.with(window.matchMedia("(prefers-color-scheme: dark)"));
/**
 * This can be used to construct a behavior to apply a prefers color scheme: light only stylesheet.
 * @public
 */

MatchMediaStyleSheetBehavior.with(window.matchMedia("(prefers-color-scheme: light)"));

/**
 * A behavior to add or remove a stylesheet from an element based on a property. The behavior ensures that
 * styles are applied while the property matches and that styles are not applied if the property does
 * not match.
 *
 * @public
 */

class PropertyStyleSheetBehavior {
  /**
   * Constructs a {@link PropertyStyleSheetBehavior} instance.
   * @param propertyName - The property name to operate from.
   * @param value - The property value to operate from.
   * @param styles - The styles to coordinate with the property.
   */
  constructor(propertyName, value, styles) {
    this.propertyName = propertyName;
    this.value = value;
    this.styles = styles;
  }
  /**
   * Binds the behavior to the element.
   * @param elementInstance - The element for which the property is applied.
   */


  bind(elementInstance) {
    Observable.getNotifier(elementInstance).subscribe(this, this.propertyName);
    this.handleChange(elementInstance, this.propertyName);
  }
  /**
   * Unbinds the behavior from the element.
   * @param source - The element for which the behavior is unbinding.
   * @internal
   */


  unbind(source) {
    Observable.getNotifier(source).unsubscribe(this, this.propertyName);
    source.$fastController.removeStyles(this.styles);
  }
  /**
   * Change event for the provided element.
   * @param source - the element for which to attach or detach styles.
   * @param key - the key to lookup to know if the element already has the styles
   * @internal
   */


  handleChange(source, key) {
    if (source[key] === this.value) {
      source.$fastController.addStyles(this.styles);
    } else {
      source.$fastController.removeStyles(this.styles);
    }
  }

}

/**
 * The CSS value for disabled cursors.
 * @public
 */
const disabledCursor = "not-allowed";

/**
 * A CSS fragment to set `display: none;` when the host is hidden using the [hidden] attribute.
 * @public
 */
const hidden = `:host([hidden]){display:none}`;
/**
 * Applies a CSS display property.
 * Also adds CSS rules to not display the element when the [hidden] attribute is applied to the element.
 * @param display - The CSS display property value
 * @public
 */

function display(displayValue) {
  return `${hidden}:host{display:${displayValue}}`;
}

/**
 * The string representing the focus selector to be used. Value
 * will be "focus-visible" when https://drafts.csswg.org/selectors-4/#the-focus-visible-pseudo
 * is supported and "focus" when it is not.
 *
 * @public
 */

const focusVisible = canUseFocusVisible() ? "focus-visible" : "focus";

/**
 * Ensures that an input number does not exceed a max value and is not less than a min value.
 * @param i - the number to clamp
 * @param min - the maximum (inclusive) value
 * @param max - the minimum (inclusive) value
 * @public
 */
function clamp(i, min, max) {
  if (isNaN(i) || i <= min) {
    return min;
  } else if (i >= max) {
    return max;
  }

  return i;
}
/**
 * Scales an input to a number between 0 and 1
 * @param i - a number between min and max
 * @param min - the max value
 * @param max - the min value
 * @public
 */

function normalize(i, min, max) {
  if (isNaN(i) || i <= min) {
    return 0.0;
  } else if (i >= max) {
    return 1.0;
  }

  return i / (max - min);
}
/**
 * Scales a number between 0 and 1
 * @param i - the number to denormalize
 * @param min - the min value
 * @param max - the max value
 * @public
 */

function denormalize(i, min, max) {
  if (isNaN(i)) {
    return min;
  }

  return min + i * (max - min);
}
/**
 * Converts degrees to radians.
 * @param i - degrees
 * @public
 */

function degreesToRadians(i) {
  return i * (Math.PI / 180.0);
}
/**
 * Converts radians to degrees.
 * @param i - radians
 * @public
 */

function radiansToDegrees(i) {
  return i * (180.0 / Math.PI);
}
/**
 * Converts a number between 0 and 255 to a hex string.
 * @param i - the number to convert to a hex string
 * @public
 */

function getHexStringForByte(i) {
  const s = Math.round(clamp(i, 0.0, 255.0)).toString(16);

  if (s.length === 1) {
    return "0" + s;
  }

  return s;
}
/**
 * Linearly interpolate
 * @public
 */

function lerp(i, min, max) {
  if (isNaN(i) || i <= 0.0) {
    return min;
  } else if (i >= 1.0) {
    return max;
  }

  return min + i * (max - min);
}
/**
 * Linearly interpolate angles in degrees
 * @public
 */

function lerpAnglesInDegrees(i, min, max) {
  if (i <= 0.0) {
    return min % 360.0;
  } else if (i >= 1.0) {
    return max % 360.0;
  }

  const a = (min - max + 360.0) % 360.0;
  const b = (max - min + 360.0) % 360.0;

  if (a <= b) {
    return (min - a * i + 360.0) % 360.0;
  }

  return (min + a * i + 360.0) % 360.0;
}
/**
 *
 * Will return infinity if i*10^(precision) overflows number
 * note that floating point rounding rules come into play here
 * so values that end up rounding on a .5 round to the nearest
 * even not always up so 2.5 rounds to 2
 * @param i - the number to round
 * @param precision - the precision to round to
 *
 * @public
 */

function roundToPrecisionSmall(i, precision) {
  const factor = Math.pow(10, precision);
  return Math.round(i * factor) / factor;
}

/**
 * This uses Hue values in "degree" format. So expect a range of [0,360]. Some other implementations instead uses radians or a normalized Hue with range [0,1]. Be aware of this when checking values or using other libraries.
 *
 * @public
 */

class ColorHSL {
  constructor(hue, sat, lum) {
    this.h = hue;
    this.s = sat;
    this.l = lum;
  }
  /**
   * Construct a {@link ColorHSL} from a config object.
   */


  static fromObject(data) {
    if (data && !isNaN(data.h) && !isNaN(data.s) && !isNaN(data.l)) {
      return new ColorHSL(data.h, data.s, data.l);
    }

    return null;
  }
  /**
   * Determines if a color is equal to another
   * @param rhs - the value to compare
   */


  equalValue(rhs) {
    return this.h === rhs.h && this.s === rhs.s && this.l === rhs.l;
  }
  /**
   * Returns a new {@link ColorHSL} rounded to the provided precision
   * @param precision - the precision to round to
   */


  roundToPrecision(precision) {
    return new ColorHSL(roundToPrecisionSmall(this.h, precision), roundToPrecisionSmall(this.s, precision), roundToPrecisionSmall(this.l, precision));
  }
  /**
   * Returns the {@link ColorHSL} formatted as an object.
   */


  toObject() {
    return {
      h: this.h,
      s: this.s,
      l: this.l
    };
  }

}

/**
 * This uses Hue values in "degree" format. So expect a range of [0,360]. Some other implementations instead uses radians or a normalized Hue with range [0,1]. Be aware of this when checking values or using other libraries.
 *
 * @public
 */

class ColorHSV {
  constructor(hue, sat, val) {
    this.h = hue;
    this.s = sat;
    this.v = val;
  }
  /**
   * Construct a {@link ColorHSV} from a config object.
   */


  static fromObject(data) {
    if (data && !isNaN(data.h) && !isNaN(data.s) && !isNaN(data.v)) {
      return new ColorHSV(data.h, data.s, data.v);
    }

    return null;
  }
  /**
   * Determines if a color is equal to another
   * @param rhs - the value to compare
   */


  equalValue(rhs) {
    return this.h === rhs.h && this.s === rhs.s && this.v === rhs.v;
  }
  /**
   * Returns a new {@link ColorHSV} rounded to the provided precision
   * @param precision - the precision to round to
   */


  roundToPrecision(precision) {
    return new ColorHSV(roundToPrecisionSmall(this.h, precision), roundToPrecisionSmall(this.s, precision), roundToPrecisionSmall(this.v, precision));
  }
  /**
   * Returns the {@link ColorHSV} formatted as an object.
   */


  toObject() {
    return {
      h: this.h,
      s: this.s,
      v: this.v
    };
  }

}

/**
 * {@link https://en.wikipedia.org/wiki/CIELAB_color_space | CIELAB color space}
 * This implementation uses the D65 constants for 2 degrees. That determines the constants used for the pure white point of the XYZ space of 0.95047, 1.0, 1.08883.
 * {@link https://en.wikipedia.org/wiki/Illuminant_D65}
 * These constants determine how the XYZ, LCH and LAB colors convert to/from RGB.
 *
 * @public
 */

class ColorLAB {
  constructor(l, a, b) {
    this.l = l;
    this.a = a;
    this.b = b;
  }
  /**
   * Construct a {@link ColorLAB} from a config object.
   */


  static fromObject(data) {
    if (data && !isNaN(data.l) && !isNaN(data.a) && !isNaN(data.b)) {
      return new ColorLAB(data.l, data.a, data.b);
    }

    return null;
  }
  /**
   * Determines if a color is equal to another
   * @param rhs - the value to compare
   */


  equalValue(rhs) {
    return this.l === rhs.l && this.a === rhs.a && this.b === rhs.b;
  }
  /**
   * Returns a new {@link ColorLAB} rounded to the provided precision
   * @param precision - the precision to round to
   */


  roundToPrecision(precision) {
    return new ColorLAB(roundToPrecisionSmall(this.l, precision), roundToPrecisionSmall(this.a, precision), roundToPrecisionSmall(this.b, precision));
  }
  /**
   * Returns the {@link ColorLAB} formatted as an object.
   */


  toObject() {
    return {
      l: this.l,
      a: this.a,
      b: this.b
    };
  }

}
ColorLAB.epsilon = 216 / 24389;
ColorLAB.kappa = 24389 / 27;

/**
 *
 * {@link https://en.wikipedia.org/wiki/CIELAB_color_space | CIELCH color space}
 *
 * This is a cylindrical representation of the CIELAB space useful for saturation operations
 * This uses Hue values in "degree" format. So expect a range of [0,360]. Some other implementations instead uses radians or a normalized Hue with range [0,1]. Be aware of this when checking values or using other libraries.
 * This implementation uses the D65 constants for 2 degrees. That determines the constants used for the pure white point of the XYZ space of 0.95047, 1.0, 1.08883.
 * {@link https://en.wikipedia.org/wiki/Illuminant_D65}
 * These constants determine how the XYZ, LCH and LAB colors convert to/from RGB.
 *
 * @public
 */

class ColorLCH {
  constructor(l, c, h) {
    this.l = l;
    this.c = c;
    this.h = h;
  }
  /**
   * Construct a {@link ColorLCH} from a config object.
   * @param data - the config object
   */


  static fromObject(data) {
    if (data && !isNaN(data.l) && !isNaN(data.c) && !isNaN(data.h)) {
      return new ColorLCH(data.l, data.c, data.h);
    }

    return null;
  }
  /**
   * Determines if one color is equal to another.
   * @param rhs - the color to compare
   */


  equalValue(rhs) {
    return this.l === rhs.l && this.c === rhs.c && this.h === rhs.h;
  }
  /**
   * Returns a new {@link ColorLCH} rounded to the provided precision
   * @param precision - the precision to round to
   */


  roundToPrecision(precision) {
    return new ColorLCH(roundToPrecisionSmall(this.l, precision), roundToPrecisionSmall(this.c, precision), roundToPrecisionSmall(this.h, precision));
  }
  /**
   * Converts the {@link ColorLCH} to a config object.
   */


  toObject() {
    return {
      l: this.l,
      c: this.c,
      h: this.h
    };
  }

}

/**
 * A RGBA color with 64 bit channels.
 *
 * @example
 * ```ts
 * new ColorRGBA64(1, 0, 0, 1) // red
 * ```
 * @public
 */

class ColorRGBA64 {
  /**
   *
   * @param red - the red value
   * @param green - the green value
   * @param blue - the blue value
   * @param alpha - the alpha value
   */
  constructor(red, green, blue, alpha) {
    this.r = red;
    this.g = green;
    this.b = blue;
    this.a = typeof alpha === "number" && !isNaN(alpha) ? alpha : 1;
  }
  /**
   * Construct a {@link ColorRGBA64} from a {@link ColorRGBA64Config}
   * @param data - the config object
   */


  static fromObject(data) {
    return data && !isNaN(data.r) && !isNaN(data.g) && !isNaN(data.b) ? new ColorRGBA64(data.r, data.g, data.b, data.a) : null;
  }
  /**
   * Determines if one color is equal to another.
   * @param rhs - the color to compare
   */


  equalValue(rhs) {
    return this.r === rhs.r && this.g === rhs.g && this.b === rhs.b && this.a === rhs.a;
  }
  /**
   * Returns the color formatted as a string; #RRGGBB
   */


  toStringHexRGB() {
    return "#" + [this.r, this.g, this.b].map(this.formatHexValue).join("");
  }
  /**
   * Returns the color formatted as a string; #RRGGBBAA
   */


  toStringHexRGBA() {
    return this.toStringHexRGB() + this.formatHexValue(this.a);
  }
  /**
   * Returns the color formatted as a string; #AARRGGBB
   */


  toStringHexARGB() {
    return "#" + [this.a, this.r, this.g, this.b].map(this.formatHexValue).join("");
  }
  /**
   * Returns the color formatted as a string; "rgb(0xRR, 0xGG, 0xBB)"
   */


  toStringWebRGB() {
    return `rgb(${Math.round(denormalize(this.r, 0.0, 255.0))},${Math.round(denormalize(this.g, 0.0, 255.0))},${Math.round(denormalize(this.b, 0.0, 255.0))})`;
  }
  /**
   * Returns the color formatted as a string; "rgba(0xRR, 0xGG, 0xBB, a)"
   * @remarks
   * Note that this follows the convention of putting alpha in the range [0.0,1.0] while the other three channels are [0,255]
   */


  toStringWebRGBA() {
    return `rgba(${Math.round(denormalize(this.r, 0.0, 255.0))},${Math.round(denormalize(this.g, 0.0, 255.0))},${Math.round(denormalize(this.b, 0.0, 255.0))},${clamp(this.a, 0, 1)})`;
  }
  /**
   * Returns a new {@link ColorRGBA64} rounded to the provided precision
   * @param precision - the precision to round to
   */


  roundToPrecision(precision) {
    return new ColorRGBA64(roundToPrecisionSmall(this.r, precision), roundToPrecisionSmall(this.g, precision), roundToPrecisionSmall(this.b, precision), roundToPrecisionSmall(this.a, precision));
  }
  /**
   * Returns a new {@link ColorRGBA64} with channel values clamped between 0 and 1.
   */


  clamp() {
    return new ColorRGBA64(clamp(this.r, 0, 1), clamp(this.g, 0, 1), clamp(this.b, 0, 1), clamp(this.a, 0, 1));
  }
  /**
   * Converts the {@link ColorRGBA64} to a {@link ColorRGBA64Config}.
   */


  toObject() {
    return {
      r: this.r,
      g: this.g,
      b: this.b,
      a: this.a
    };
  }

  formatHexValue(value) {
    return getHexStringForByte(denormalize(value, 0.0, 255.0));
  }

}

/**
 * {@link https://en.wikipedia.org/wiki/CIE_1931_color_space | XYZ color space}
 *
 * This implementation uses the D65 constants for 2 degrees. That determines the constants used for the pure white point of the XYZ space of 0.95047, 1.0, 1.08883.
 * {@link https://en.wikipedia.org/wiki/Illuminant_D65}
 * These constants determine how the XYZ, LCH and LAB colors convert to/from RGB.
 *
 * @public
 */

class ColorXYZ {
  constructor(x, y, z) {
    this.x = x;
    this.y = y;
    this.z = z;
  }
  /**
   * Construct a {@link ColorXYZ} from a config object.
   */


  static fromObject(data) {
    if (data && !isNaN(data.x) && !isNaN(data.y) && !isNaN(data.z)) {
      return new ColorXYZ(data.x, data.y, data.z);
    }

    return null;
  }
  /**
   * Determines if a color is equal to another
   * @param rhs - the value to compare
   */


  equalValue(rhs) {
    return this.x === rhs.x && this.y === rhs.y && this.z === rhs.z;
  }
  /**
   * Returns a new {@link ColorXYZ} rounded to the provided precision
   * @param precision - the precision to round to
   */


  roundToPrecision(precision) {
    return new ColorXYZ(roundToPrecisionSmall(this.x, precision), roundToPrecisionSmall(this.y, precision), roundToPrecisionSmall(this.z, precision));
  }
  /**
   * Returns the {@link ColorXYZ} formatted as an object.
   */


  toObject() {
    return {
      x: this.x,
      y: this.y,
      z: this.z
    };
  }

}
/**
 * D65 2 degree white point
 */

ColorXYZ.whitePoint = new ColorXYZ(0.95047, 1.0, 1.08883);

// All conversions use the D65 2 degree white point for XYZ
// Info on conversions and constants used can be found in the following:
// https://en.wikipedia.org/wiki/CIELAB_color_space
// https://en.wikipedia.org/wiki/Illuminant_D65
// https://ninedegreesbelow.com/photography/xyz-rgb.html
// http://user.engineering.uiowa.edu/~aip/Misc/ColorFAQ.html
// https://web.stanford.edu/~sujason/ColorBalancing/adaptation.html
// http://brucelindbloom.com/index.html

/**
 * Get the luminance of a color in the linear RGB space.
 * This is not the same as the relative luminance in the sRGB space for WCAG contrast calculations. Use rgbToRelativeLuminance instead.
 * @param rgb - The input color
 *
 * @public
 */

function rgbToLinearLuminance(rgb) {
  return rgb.r * 0.2126 + rgb.g * 0.7152 + rgb.b * 0.0722;
}
/**
 * Get the relative luminance of a color.
 * Adjusts the color to sRGB space, which is necessary for the WCAG contrast spec.
 * The alpha channel of the input is ignored.
 * @param rgb - The input color
 *
 * @public
 */

function rgbToRelativeLuminance(rgb) {
  function luminanceHelper(i) {
    if (i <= 0.03928) {
      return i / 12.92;
    }

    return Math.pow((i + 0.055) / 1.055, 2.4);
  }

  return rgbToLinearLuminance(new ColorRGBA64(luminanceHelper(rgb.r), luminanceHelper(rgb.g), luminanceHelper(rgb.b), 1));
}

const calculateContrastRatio = (a, b) => (a + 0.05) / (b + 0.05);
/**
 * Calculate the contrast ratio between two colors. Uses the formula described by {@link https://www.w3.org/TR/WCAG20-TECHS/G17.html | WCAG 2.0}.
 *
 * @remarks
 * The alpha channel of the input is ignored
 *
 * @public
 */


function contrastRatio(a, b) {
  const luminanceA = rgbToRelativeLuminance(a);
  const luminanceB = rgbToRelativeLuminance(b);
  return luminanceA > luminanceB ? calculateContrastRatio(luminanceA, luminanceB) : calculateContrastRatio(luminanceB, luminanceA);
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorRGBA64} to a {@link @microsoft/fast-colors#ColorHSL}
 * @param rgb - the rgb color to convert
 *
 * @remarks
 * The alpha channel of the input is ignored
 *
 * @public
 */

function rgbToHSL(rgb) {
  const max = Math.max(rgb.r, rgb.g, rgb.b);
  const min = Math.min(rgb.r, rgb.g, rgb.b);
  const delta = max - min;
  let hue = 0;

  if (delta !== 0) {
    if (max === rgb.r) {
      hue = 60 * ((rgb.g - rgb.b) / delta % 6);
    } else if (max === rgb.g) {
      hue = 60 * ((rgb.b - rgb.r) / delta + 2);
    } else {
      hue = 60 * ((rgb.r - rgb.g) / delta + 4);
    }
  }

  if (hue < 0) {
    hue += 360;
  }

  const lum = (max + min) / 2;
  let sat = 0;

  if (delta !== 0) {
    sat = delta / (1 - Math.abs(2 * lum - 1));
  }

  return new ColorHSL(hue, sat, lum);
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorHSL} to a {@link @microsoft/fast-colors#ColorRGBA64}
 * @param hsl - the hsl color to convert
 * @param alpha - the alpha value
 *
 * @public
 */

function hslToRGB(hsl, alpha = 1) {
  const c = (1 - Math.abs(2 * hsl.l - 1)) * hsl.s;
  const x = c * (1 - Math.abs(hsl.h / 60 % 2 - 1));
  const m = hsl.l - c / 2;
  let r = 0;
  let g = 0;
  let b = 0;

  if (hsl.h < 60) {
    r = c;
    g = x;
    b = 0;
  } else if (hsl.h < 120) {
    r = x;
    g = c;
    b = 0;
  } else if (hsl.h < 180) {
    r = 0;
    g = c;
    b = x;
  } else if (hsl.h < 240) {
    r = 0;
    g = x;
    b = c;
  } else if (hsl.h < 300) {
    r = x;
    g = 0;
    b = c;
  } else if (hsl.h < 360) {
    r = c;
    g = 0;
    b = x;
  }

  return new ColorRGBA64(r + m, g + m, b + m, alpha);
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorRGBA64} to a {@link @microsoft/fast-colors#ColorHSV}
 * @param rgb - the rgb color to convert
 *
 * @remarks
 * The alpha channel of the input is ignored
 *
 * @public
 */

function rgbToHSV(rgb) {
  const max = Math.max(rgb.r, rgb.g, rgb.b);
  const min = Math.min(rgb.r, rgb.g, rgb.b);
  const delta = max - min;
  let hue = 0;

  if (delta !== 0) {
    if (max === rgb.r) {
      hue = 60 * ((rgb.g - rgb.b) / delta % 6);
    } else if (max === rgb.g) {
      hue = 60 * ((rgb.b - rgb.r) / delta + 2);
    } else {
      hue = 60 * ((rgb.r - rgb.g) / delta + 4);
    }
  }

  if (hue < 0) {
    hue += 360;
  }

  let sat = 0;

  if (max !== 0) {
    sat = delta / max;
  }

  return new ColorHSV(hue, sat, max);
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorHSV} to a {@link @microsoft/fast-colors#ColorRGBA64}
 * @param hsv - the hsv color to convert
 * @param alpha - the alpha value
 *
 * @public
 */

function hsvToRGB(hsv, alpha = 1) {
  const c = hsv.s * hsv.v;
  const x = c * (1 - Math.abs(hsv.h / 60 % 2 - 1));
  const m = hsv.v - c;
  let r = 0;
  let g = 0;
  let b = 0;

  if (hsv.h < 60) {
    r = c;
    g = x;
    b = 0;
  } else if (hsv.h < 120) {
    r = x;
    g = c;
    b = 0;
  } else if (hsv.h < 180) {
    r = 0;
    g = c;
    b = x;
  } else if (hsv.h < 240) {
    r = 0;
    g = x;
    b = c;
  } else if (hsv.h < 300) {
    r = x;
    g = 0;
    b = c;
  } else if (hsv.h < 360) {
    r = c;
    g = 0;
    b = x;
  }

  return new ColorRGBA64(r + m, g + m, b + m, alpha);
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorLCH} to a {@link @microsoft/fast-colors#ColorLAB}
 * @param lch - the lch color to convert
 *
 * @public
 */

function lchToLAB(lch) {
  let a = 0;
  let b = 0;

  if (lch.h !== 0) {
    a = Math.cos(degreesToRadians(lch.h)) * lch.c;
    b = Math.sin(degreesToRadians(lch.h)) * lch.c;
  }

  return new ColorLAB(lch.l, a, b);
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorLAB} to a {@link @microsoft/fast-colors#ColorLCH}
 * @param lab - the lab color to convert
 *
 * @remarks
 * The discontinuity in the C parameter at 0 means that floating point errors will often result in values near 0 giving unpredictable results.
 * EG: 0.0000001 gives a very different result than -0.0000001
 * In cases where both a and b are very near zero this function will return an LCH color with an H of 0
 * More info about the atan2 function: {@link https://en.wikipedia.org/wiki/Atan2}
 * @public
 */

function labToLCH(lab) {
  let h = 0; // Because of the discontinuity at 0 if a number is very close to 0 - often due to floating point errors - then
  // it gives unexpected results. EG: 0.000000000001 gives a different result than 0. So just avoid any number
  // that has both a and b very close to zero and lump it in with the h = 0 case.

  if (Math.abs(lab.b) > 0.001 || Math.abs(lab.a) > 0.001) {
    h = radiansToDegrees(Math.atan2(lab.b, lab.a));
  }

  if (h < 0) {
    h += 360;
  }

  const c = Math.sqrt(lab.a * lab.a + lab.b * lab.b);
  return new ColorLCH(lab.l, c, h);
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorLAB} to a {@link @microsoft/fast-colors#ColorXYZ}
 * @param lab - the lab color to convert
 *
 * @public
 */

function labToXYZ(lab) {
  const fy = (lab.l + 16) / 116;
  const fx = fy + lab.a / 500;
  const fz = fy - lab.b / 200;
  const xcubed = Math.pow(fx, 3);
  const ycubed = Math.pow(fy, 3);
  const zcubed = Math.pow(fz, 3);
  let x = 0;

  if (xcubed > ColorLAB.epsilon) {
    x = xcubed;
  } else {
    x = (116 * fx - 16) / ColorLAB.kappa;
  }

  let y = 0;

  if (lab.l > ColorLAB.epsilon * ColorLAB.kappa) {
    y = ycubed;
  } else {
    y = lab.l / ColorLAB.kappa;
  }

  let z = 0;

  if (zcubed > ColorLAB.epsilon) {
    z = zcubed;
  } else {
    z = (116 * fz - 16) / ColorLAB.kappa;
  }

  x = ColorXYZ.whitePoint.x * x;
  y = ColorXYZ.whitePoint.y * y;
  z = ColorXYZ.whitePoint.z * z;
  return new ColorXYZ(x, y, z);
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorXYZ} to a {@link @microsoft/fast-colors#ColorLAB}
 * @param xyz - the xyz color to convert
 *
 * @public
 */

function xyzToLAB(xyz) {
  function xyzToLABHelper(i) {
    if (i > ColorLAB.epsilon) {
      return Math.pow(i, 1 / 3);
    }

    return (ColorLAB.kappa * i + 16) / 116;
  }

  const x = xyzToLABHelper(xyz.x / ColorXYZ.whitePoint.x);
  const y = xyzToLABHelper(xyz.y / ColorXYZ.whitePoint.y);
  const z = xyzToLABHelper(xyz.z / ColorXYZ.whitePoint.z);
  const l = 116 * y - 16;
  const a = 500 * (x - y);
  const b = 200 * (y - z);
  return new ColorLAB(l, a, b);
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorRGBA64} to a {@link @microsoft/fast-colors#ColorXYZ}
 * @param rgb - the rgb color to convert
 *
 * @remarks
 * The alpha channel of the input is ignored
 * @public
 */

function rgbToXYZ(rgb) {
  function rgbToXYZHelper(i) {
    if (i <= 0.04045) {
      return i / 12.92;
    }

    return Math.pow((i + 0.055) / 1.055, 2.4);
  }

  const r = rgbToXYZHelper(rgb.r);
  const g = rgbToXYZHelper(rgb.g);
  const b = rgbToXYZHelper(rgb.b);
  const x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
  const y = r * 0.2126729 + g * 0.7151522 + b * 0.072175;
  const z = r * 0.0193339 + g * 0.119192 + b * 0.9503041;
  return new ColorXYZ(x, y, z);
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorXYZ} to a {@link @microsoft/fast-colors#ColorRGBA64}
 * @param xyz - the xyz color to convert
 * @param alpha - the alpha value
 *
 * @remarks
 * Note that the xyz color space is significantly larger than sRGB. As such, this can return colors rgb values greater than 1 or less than 0
 * @public
 */

function xyzToRGB(xyz, alpha = 1) {
  function xyzToRGBHelper(i) {
    if (i <= 0.0031308) {
      return i * 12.92;
    }

    return 1.055 * Math.pow(i, 1 / 2.4) - 0.055;
  }

  const r = xyzToRGBHelper(xyz.x * 3.2404542 - xyz.y * 1.5371385 - xyz.z * 0.4985314);
  const g = xyzToRGBHelper(xyz.x * -0.969266 + xyz.y * 1.8760108 + xyz.z * 0.041556);
  const b = xyzToRGBHelper(xyz.x * 0.0556434 - xyz.y * 0.2040259 + xyz.z * 1.0572252);
  return new ColorRGBA64(r, g, b, alpha);
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorRGBA64} to a {@link @microsoft/fast-colors#ColorLAB}
 * @param rgb - the rgb color to convert
 *
 * @remarks
 * The alpha channel of the input is ignored
 *
 * @public
 */

function rgbToLAB(rgb) {
  return xyzToLAB(rgbToXYZ(rgb));
}
/**
 * Converts a {@link @microsoft/fast-colors#ColorLAB} to a {@link @microsoft/fast-colors#ColorRGBA64}
 * @param lab - the LAB color to convert
 * @param alpha - the alpha value
 *
 * @remarks
 * Note that the xyz color space (which the conversion from LAB uses) is significantly larger than sRGB. As such, this can return colors rgb values greater than 1 or less than 0
 *
 * @public
 */

function labToRGB(lab, alpha = 1) {
  return xyzToRGB(labToXYZ(lab), alpha);
}
/**
 * Convert a {@link @microsoft/fast-colors#ColorRGBA64} to a {@link @microsoft/fast-colors#ColorLCH}
 *
 * @param rgb - the rgb color to convert
 *
 * @remarks
 * The alpha channel of the input is ignored
 *
 * @public
 */

function rgbToLCH(rgb) {
  return labToLCH(rgbToLAB(rgb));
}
/**
 * Convert a {@link @microsoft/fast-colors#ColorLCH} to a {@link @microsoft/fast-colors#ColorRGBA64}
 * @param lch - the LCH color to convert
 * @param alpha - the alpha value
 *
 * @public
 */

function lchToRGB(lch, alpha = 1) {
  return labToRGB(lchToLAB(lch), alpha);
}

/**
 * Saturate a color using LCH color space
 *
 * @remarks
 * The alpha channel of the input is ignored
 *
 * @public
 */

function saturateViaLCH(input, saturation, saturationConstant = 18) {
  const lch = rgbToLCH(input);
  let sat = lch.c + saturation * saturationConstant;

  if (sat < 0) {
    sat = 0;
  }

  return lchToRGB(new ColorLCH(lch.l, sat, lch.h));
}
/**
 * @public
 */

function blendMultiplyChannel(bottom, top) {
  return bottom * top;
}
/**
 * Blends two colors with the multiply mode
 *
 * @remarks
 * The alpha channel of the input is ignored
 *
 * @public
 */

function blendMultiply(bottom, top) {
  return new ColorRGBA64(blendMultiplyChannel(bottom.r, top.r), blendMultiplyChannel(bottom.g, top.g), blendMultiplyChannel(bottom.b, top.b), 1);
}
/**
 * @public
 */

function blendOverlayChannel(bottom, top) {
  if (bottom < 0.5) {
    return clamp(2.0 * top * bottom, 0, 1);
  }

  return clamp(1.0 - 2.0 * (1.0 - top) * (1.0 - bottom), 0, 1);
}
/**
 * Blends two colors with the overlay mode
 *
 * @remarks
 * The alpha channel of the input is ignored
 *
 * @public
 */

function blendOverlay(bottom, top) {
  return new ColorRGBA64(blendOverlayChannel(bottom.r, top.r), blendOverlayChannel(bottom.g, top.g), blendOverlayChannel(bottom.b, top.b), 1);
}
/**
 * Color blend modes.
 * @public
 */

var ColorBlendMode;

(function (ColorBlendMode) {
  ColorBlendMode[ColorBlendMode["Burn"] = 0] = "Burn";
  ColorBlendMode[ColorBlendMode["Color"] = 1] = "Color";
  ColorBlendMode[ColorBlendMode["Darken"] = 2] = "Darken";
  ColorBlendMode[ColorBlendMode["Dodge"] = 3] = "Dodge";
  ColorBlendMode[ColorBlendMode["Lighten"] = 4] = "Lighten";
  ColorBlendMode[ColorBlendMode["Multiply"] = 5] = "Multiply";
  ColorBlendMode[ColorBlendMode["Overlay"] = 6] = "Overlay";
  ColorBlendMode[ColorBlendMode["Screen"] = 7] = "Screen";
})(ColorBlendMode || (ColorBlendMode = {}));

/**
 * Interpolate by RGB color space
 *
 * @public
 */

function interpolateRGB(position, left, right) {
  if (isNaN(position) || position <= 0) {
    return left;
  } else if (position >= 1) {
    return right;
  }

  return new ColorRGBA64(lerp(position, left.r, right.r), lerp(position, left.g, right.g), lerp(position, left.b, right.b), lerp(position, left.a, right.a));
}
/**
 * Interpolate by HSL color space
 *
 * @public
 */

function interpolateHSL(position, left, right) {
  if (isNaN(position) || position <= 0) {
    return left;
  } else if (position >= 1) {
    return right;
  }

  return new ColorHSL(lerpAnglesInDegrees(position, left.h, right.h), lerp(position, left.s, right.s), lerp(position, left.l, right.l));
}
/**
 * Interpolate by HSV color space
 *
 * @public
 */

function interpolateHSV(position, left, right) {
  if (isNaN(position) || position <= 0) {
    return left;
  } else if (position >= 1) {
    return right;
  }

  return new ColorHSV(lerpAnglesInDegrees(position, left.h, right.h), lerp(position, left.s, right.s), lerp(position, left.v, right.v));
}
/**
 * Interpolate by XYZ color space
 *
 * @public
 */

function interpolateXYZ(position, left, right) {
  if (isNaN(position) || position <= 0) {
    return left;
  } else if (position >= 1) {
    return right;
  }

  return new ColorXYZ(lerp(position, left.x, right.x), lerp(position, left.y, right.y), lerp(position, left.z, right.z));
}
/**
 * Interpolate by LAB color space
 *
 * @public
 */

function interpolateLAB(position, left, right) {
  if (isNaN(position) || position <= 0) {
    return left;
  } else if (position >= 1) {
    return right;
  }

  return new ColorLAB(lerp(position, left.l, right.l), lerp(position, left.a, right.a), lerp(position, left.b, right.b));
}
/**
 * Interpolate by LCH color space
 *
 * @public
 */

function interpolateLCH(position, left, right) {
  if (isNaN(position) || position <= 0) {
    return left;
  } else if (position >= 1) {
    return right;
  }

  return new ColorLCH(lerp(position, left.l, right.l), lerp(position, left.c, right.c), lerpAnglesInDegrees(position, left.h, right.h));
}
/**
 * Color interpolation spaces
 *
 * @public
 */

var ColorInterpolationSpace;

(function (ColorInterpolationSpace) {
  ColorInterpolationSpace[ColorInterpolationSpace["RGB"] = 0] = "RGB";
  ColorInterpolationSpace[ColorInterpolationSpace["HSL"] = 1] = "HSL";
  ColorInterpolationSpace[ColorInterpolationSpace["HSV"] = 2] = "HSV";
  ColorInterpolationSpace[ColorInterpolationSpace["XYZ"] = 3] = "XYZ";
  ColorInterpolationSpace[ColorInterpolationSpace["LAB"] = 4] = "LAB";
  ColorInterpolationSpace[ColorInterpolationSpace["LCH"] = 5] = "LCH";
})(ColorInterpolationSpace || (ColorInterpolationSpace = {}));
/**
 * Interpolate by color space
 *
 * @public
 */


function interpolateByColorSpace(position, space, left, right) {
  if (isNaN(position) || position <= 0) {
    return left;
  } else if (position >= 1) {
    return right;
  }

  switch (space) {
    case ColorInterpolationSpace.HSL:
      return hslToRGB(interpolateHSL(position, rgbToHSL(left), rgbToHSL(right)));

    case ColorInterpolationSpace.HSV:
      return hsvToRGB(interpolateHSV(position, rgbToHSV(left), rgbToHSV(right)));

    case ColorInterpolationSpace.XYZ:
      return xyzToRGB(interpolateXYZ(position, rgbToXYZ(left), rgbToXYZ(right)));

    case ColorInterpolationSpace.LAB:
      return labToRGB(interpolateLAB(position, rgbToLAB(left), rgbToLAB(right)));

    case ColorInterpolationSpace.LCH:
      return lchToRGB(interpolateLCH(position, rgbToLCH(left), rgbToLCH(right)));

    default:
      return interpolateRGB(position, left, right);
  }
}

/**
 * A color scale created from linear stops
 * @public
 */

class ColorScale {
  constructor(stops) {
    if (stops == null || stops.length === 0) {
      throw new Error("The stops argument must be non-empty");
    } else {
      this.stops = this.sortColorScaleStops(stops);
    }
  }

  static createBalancedColorScale(colors) {
    if (colors == null || colors.length === 0) {
      throw new Error("The colors argument must be non-empty");
    }

    const stops = new Array(colors.length);

    for (let i = 0; i < colors.length; i++) {
      // Special case first and last in order to avoid floating point jaggies
      if (i === 0) {
        stops[i] = {
          color: colors[i],
          position: 0
        };
      } else if (i === colors.length - 1) {
        stops[i] = {
          color: colors[i],
          position: 1
        };
      } else {
        stops[i] = {
          color: colors[i],
          position: i * (1 / (colors.length - 1))
        };
      }
    }

    return new ColorScale(stops);
  }

  getColor(position, interpolationMode = ColorInterpolationSpace.RGB) {
    if (this.stops.length === 1) {
      return this.stops[0].color;
    } else if (position <= 0) {
      return this.stops[0].color;
    } else if (position >= 1) {
      return this.stops[this.stops.length - 1].color;
    }

    let lowerIndex = 0;

    for (let i = 0; i < this.stops.length; i++) {
      if (this.stops[i].position <= position) {
        lowerIndex = i;
      }
    }

    let upperIndex = lowerIndex + 1;

    if (upperIndex >= this.stops.length) {
      upperIndex = this.stops.length - 1;
    }

    const scalePosition = (position - this.stops[lowerIndex].position) * (1.0 / (this.stops[upperIndex].position - this.stops[lowerIndex].position));
    return interpolateByColorSpace(scalePosition, interpolationMode, this.stops[lowerIndex].color, this.stops[upperIndex].color);
  }

  trim(lowerBound, upperBound, interpolationMode = ColorInterpolationSpace.RGB) {
    if (lowerBound < 0 || upperBound > 1 || upperBound < lowerBound) {
      throw new Error("Invalid bounds");
    }

    if (lowerBound === upperBound) {
      return new ColorScale([{
        color: this.getColor(lowerBound, interpolationMode),
        position: 0
      }]);
    }

    const containedStops = [];

    for (let i = 0; i < this.stops.length; i++) {
      if (this.stops[i].position >= lowerBound && this.stops[i].position <= upperBound) {
        containedStops.push(this.stops[i]);
      }
    }

    if (containedStops.length === 0) {
      return new ColorScale([{
        color: this.getColor(lowerBound),
        position: lowerBound
      }, {
        color: this.getColor(upperBound),
        position: upperBound
      }]);
    }

    if (containedStops[0].position !== lowerBound) {
      containedStops.unshift({
        color: this.getColor(lowerBound),
        position: lowerBound
      });
    }

    if (containedStops[containedStops.length - 1].position !== upperBound) {
      containedStops.push({
        color: this.getColor(upperBound),
        position: upperBound
      });
    }

    const range = upperBound - lowerBound;
    const finalStops = new Array(containedStops.length);

    for (let i = 0; i < containedStops.length; i++) {
      finalStops[i] = {
        color: containedStops[i].color,
        position: (containedStops[i].position - lowerBound) / range
      };
    }

    return new ColorScale(finalStops);
  }

  findNextColor(position, contrast, searchDown = false, interpolationMode = ColorInterpolationSpace.RGB, contrastErrorMargin = 0.005, maxSearchIterations = 32) {
    if (isNaN(position) || position <= 0) {
      position = 0;
    } else if (position >= 1) {
      position = 1;
    }

    const startingColor = this.getColor(position, interpolationMode);
    const finalPosition = searchDown ? 0 : 1;
    const finalColor = this.getColor(finalPosition, interpolationMode);
    const finalContrast = contrastRatio(startingColor, finalColor);

    if (finalContrast <= contrast) {
      return finalPosition;
    }

    let testRangeMin = searchDown ? 0 : position;
    let testRangeMax = searchDown ? position : 0;
    let mid = finalPosition;
    let iterations = 0;

    while (iterations <= maxSearchIterations) {
      mid = Math.abs(testRangeMax - testRangeMin) / 2 + testRangeMin;
      const midColor = this.getColor(mid, interpolationMode);
      const midContrast = contrastRatio(startingColor, midColor);

      if (Math.abs(midContrast - contrast) <= contrastErrorMargin) {
        return mid;
      } else if (midContrast > contrast) {
        if (searchDown) {
          testRangeMin = mid;
        } else {
          testRangeMax = mid;
        }
      } else {
        if (searchDown) {
          testRangeMax = mid;
        } else {
          testRangeMin = mid;
        }
      }

      iterations++;
    }

    return mid;
  }

  clone() {
    const newStops = new Array(this.stops.length);

    for (let i = 0; i < newStops.length; i++) {
      newStops[i] = {
        color: this.stops[i].color,
        position: this.stops[i].position
      };
    }

    return new ColorScale(newStops);
  }

  sortColorScaleStops(stops) {
    return stops.sort((a, b) => {
      const A = a.position;
      const B = b.position;

      if (A < B) {
        return -1;
      } else if (A > B) {
        return 1;
      } else {
        return 0;
      }
    });
  }

}

const hexRGBRegex = /^#((?:[0-9a-f]{6}|[0-9a-f]{3}))$/i; // Matches #RGB and #RRGGBBAA, where R, G, B, and A are [0-9] or [A-F]
/**
 * Converts a hexadecimal color string to a {@link @microsoft/fast-colors#ColorRGBA64}.
 * @param raw - a color string in the form of "#RRGGBB" or "#RGB"
 * @example
 * ```ts
 * parseColorHexRGBA("#FF0000");
 * parseColorHexRGBA("#F00");
 * ```
 * @public
 */

function parseColorHexRGB(raw) {
  const result = hexRGBRegex.exec(raw);

  if (result === null) {
    return null;
  }

  let digits = result[1];

  if (digits.length === 3) {
    const r = digits.charAt(0);
    const g = digits.charAt(1);
    const b = digits.charAt(2);
    digits = r.concat(r, g, g, b, b);
  }

  const rawInt = parseInt(digits, 16);

  if (isNaN(rawInt)) {
    return null;
  } // Note the use of >>> rather than >> as we want JS to manipulate these as unsigned numbers


  return new ColorRGBA64(normalize((rawInt & 0xff0000) >>> 16, 0, 255), normalize((rawInt & 0x00ff00) >>> 8, 0, 255), normalize(rawInt & 0x0000ff, 0, 255), 1);
}

/**
 * Generates a color palette
 * @public
 */

class ColorPalette {
  constructor(config) {
    this.config = Object.assign({}, ColorPalette.defaultPaletteConfig, config);
    this.palette = [];
    this.updatePaletteColors();
  }

  updatePaletteGenerationValues(newConfig) {
    let changed = false;

    for (const key in newConfig) {
      if (this.config[key]) {
        if (this.config[key].equalValue) {
          if (!this.config[key].equalValue(newConfig[key])) {
            this.config[key] = newConfig[key];
            changed = true;
          }
        } else {
          if (newConfig[key] !== this.config[key]) {
            this.config[key] = newConfig[key];
            changed = true;
          }
        }
      }
    }

    if (changed) {
      this.updatePaletteColors();
    }

    return changed;
  }

  updatePaletteColors() {
    const scale = this.generatePaletteColorScale();

    for (let i = 0; i < this.config.steps; i++) {
      this.palette[i] = scale.getColor(i / (this.config.steps - 1), this.config.interpolationMode);
    }
  }

  generatePaletteColorScale() {
    // Even when config.baseScalePosition is specified, using 0.5 for the baseColor
    // in the baseScale gives better results. Otherwise very off-center palettes
    // tend to go completely grey at the end furthest from the specified base color.
    const baseColorHSL = rgbToHSL(this.config.baseColor);
    const baseScale = new ColorScale([{
      position: 0,
      color: this.config.scaleColorLight
    }, {
      position: 0.5,
      color: this.config.baseColor
    }, {
      position: 1,
      color: this.config.scaleColorDark
    }]);
    const trimmedScale = baseScale.trim(this.config.clipLight, 1 - this.config.clipDark);
    const trimmedLight = trimmedScale.getColor(0);
    const trimmedDark = trimmedScale.getColor(1);
    let adjustedLight = trimmedLight;
    let adjustedDark = trimmedDark;

    if (baseColorHSL.s >= this.config.saturationAdjustmentCutoff) {
      adjustedLight = saturateViaLCH(adjustedLight, this.config.saturationLight);
      adjustedDark = saturateViaLCH(adjustedDark, this.config.saturationDark);
    }

    if (this.config.multiplyLight !== 0) {
      const multiply = blendMultiply(this.config.baseColor, adjustedLight);
      adjustedLight = interpolateByColorSpace(this.config.multiplyLight, this.config.interpolationMode, adjustedLight, multiply);
    }

    if (this.config.multiplyDark !== 0) {
      const multiply = blendMultiply(this.config.baseColor, adjustedDark);
      adjustedDark = interpolateByColorSpace(this.config.multiplyDark, this.config.interpolationMode, adjustedDark, multiply);
    }

    if (this.config.overlayLight !== 0) {
      const overlay = blendOverlay(this.config.baseColor, adjustedLight);
      adjustedLight = interpolateByColorSpace(this.config.overlayLight, this.config.interpolationMode, adjustedLight, overlay);
    }

    if (this.config.overlayDark !== 0) {
      const overlay = blendOverlay(this.config.baseColor, adjustedDark);
      adjustedDark = interpolateByColorSpace(this.config.overlayDark, this.config.interpolationMode, adjustedDark, overlay);
    }

    if (this.config.baseScalePosition) {
      if (this.config.baseScalePosition <= 0) {
        return new ColorScale([{
          position: 0,
          color: this.config.baseColor
        }, {
          position: 1,
          color: adjustedDark.clamp()
        }]);
      } else if (this.config.baseScalePosition >= 1) {
        return new ColorScale([{
          position: 0,
          color: adjustedLight.clamp()
        }, {
          position: 1,
          color: this.config.baseColor
        }]);
      }

      return new ColorScale([{
        position: 0,
        color: adjustedLight.clamp()
      }, {
        position: this.config.baseScalePosition,
        color: this.config.baseColor
      }, {
        position: 1,
        color: adjustedDark.clamp()
      }]);
    }

    return new ColorScale([{
      position: 0,
      color: adjustedLight.clamp()
    }, {
      position: 0.5,
      color: this.config.baseColor
    }, {
      position: 1,
      color: adjustedDark.clamp()
    }]);
  }

}
ColorPalette.defaultPaletteConfig = {
  baseColor: parseColorHexRGB("#808080"),
  steps: 11,
  interpolationMode: ColorInterpolationSpace.RGB,
  scaleColorLight: new ColorRGBA64(1, 1, 1, 1),
  scaleColorDark: new ColorRGBA64(0, 0, 0, 1),
  clipLight: 0.185,
  clipDark: 0.16,
  saturationAdjustmentCutoff: 0.05,
  saturationLight: 0.35,
  saturationDark: 1.25,
  overlayLight: 0,
  overlayDark: 0.25,
  multiplyLight: 0,
  multiplyDark: 0,
  baseScalePosition: 0.5
};
ColorPalette.greyscalePaletteConfig = {
  baseColor: parseColorHexRGB("#808080"),
  steps: 11,
  interpolationMode: ColorInterpolationSpace.RGB,
  scaleColorLight: new ColorRGBA64(1, 1, 1, 1),
  scaleColorDark: new ColorRGBA64(0, 0, 0, 1),
  clipLight: 0,
  clipDark: 0,
  saturationAdjustmentCutoff: 0,
  saturationLight: 0,
  saturationDark: 0,
  overlayLight: 0,
  overlayDark: 0,
  multiplyLight: 0,
  multiplyDark: 0,
  baseScalePosition: 0.5
};
/**
 * @public
 */

({
  targetSize: 63,
  spacing: 4,
  scaleColorLight: ColorPalette.defaultPaletteConfig.scaleColorLight,
  scaleColorDark: ColorPalette.defaultPaletteConfig.scaleColorDark
});

/**
 * Creates a color palette for UI components
 * @public
 */

class ComponentStateColorPalette {
  constructor(config) {
    this.palette = [];
    this.config = Object.assign({}, ComponentStateColorPalette.defaultPaletteConfig, config);
    this.regenPalettes();
  }

  regenPalettes() {
    let steps = this.config.steps;

    if (isNaN(steps) || steps < 3) {
      steps = 3;
    } // This palette is tuned to go as dark as differences between the levels can be perceived according to tests
    // on numerous monitors in different conditions. Stay linear from white until this first cutoff.


    const darkLum = 0.14; // In the dark compression, this is the last luminance value before full black.

    const darkestLum = 0.06; // The Color for the luminance value above, placed on the ramp at it's normal position, so darker colors after
    // it can be compressed.

    const darkLumColor = new ColorRGBA64(darkLum, darkLum, darkLum, 1); // The number of steps in the ramp that has been tuned for default use. This coincides with the size of the
    // default ramp, but the palette could be generated with fewer steps to increase final contrast. This number
    // should however stay the same.

    const stepsForLuminanceRamp = 94; // Create the reference, dark-compressed, grey palette, like:
    // F------------------------------------------------------------------------------------[dark]------[darkest]0
    //                                                                                      |--compressed area--|

    const r = new ColorPalette(Object.assign(Object.assign({}, ColorPalette.greyscalePaletteConfig), {
      baseColor: darkLumColor,
      baseScalePosition: (1 - darkLum) * 100 / stepsForLuminanceRamp,
      steps
    }));
    const referencePalette = r.palette; // Find the requested base color on the adjusted luminance reference ramp.
    // There is no _right_ way to desaturate a color, and both methods we've tested have value, so average them out.

    const baseColorLum1 = rgbToLinearLuminance(this.config.baseColor);
    const baseColorLum2 = rgbToHSL(this.config.baseColor).l;
    const baseColorLum = (baseColorLum1 + baseColorLum2) / 2;
    const baseColorRefIndex = this.matchRelativeLuminanceIndex(baseColorLum, referencePalette);
    const baseColorPercent = baseColorRefIndex / (steps - 1); // Find the luminance location for the dark cutoff.

    const darkRefIndex = this.matchRelativeLuminanceIndex(darkLum, referencePalette);
    const darkPercent = darkRefIndex / (steps - 1); // Issue https://github.com/microsoft/fast/issues/1904
    // Creating a color from H, S, and a known L value is not the inverse of getting the relative
    // luminace as above. Need to derive a relative luminance version of the color to better match on the dark end.
    // Find the dark cutoff and darkest variations of the requested base color.

    const baseColorHSL = rgbToHSL(this.config.baseColor);
    const darkBaseColor = hslToRGB(ColorHSL.fromObject({
      h: baseColorHSL.h,
      s: baseColorHSL.s,
      l: darkLum
    }));
    const darkestBaseColor = hslToRGB(ColorHSL.fromObject({
      h: baseColorHSL.h,
      s: baseColorHSL.s,
      l: darkestLum
    })); // Create the gradient stops, including the base color and anchor colors for the dark end compression.

    const fullColorScaleStops = new Array(5);
    fullColorScaleStops[0] = {
      position: 0,
      color: new ColorRGBA64(1, 1, 1, 1)
    };
    fullColorScaleStops[1] = {
      position: baseColorPercent,
      color: this.config.baseColor
    };
    fullColorScaleStops[2] = {
      position: darkPercent,
      color: darkBaseColor
    };
    fullColorScaleStops[3] = {
      position: 0.99,
      color: darkestBaseColor
    };
    fullColorScaleStops[4] = {
      position: 1,
      color: new ColorRGBA64(0, 0, 0, 1)
    };
    const scale = new ColorScale(fullColorScaleStops); // Create the palette.

    this.palette = new Array(steps);

    for (let i = 0; i < steps; i++) {
      const c = scale.getColor(i / (steps - 1), ColorInterpolationSpace.RGB);
      this.palette[i] = c;
    }
  }

  matchRelativeLuminanceIndex(input, reference) {
    let bestFitValue = Number.MAX_VALUE;
    let bestFitIndex = 0;
    let i = 0;
    const referenceLength = reference.length;

    for (; i < referenceLength; i++) {
      const fitValue = Math.abs(rgbToLinearLuminance(reference[i]) - input);

      if (fitValue < bestFitValue) {
        bestFitValue = fitValue;
        bestFitIndex = i;
      }
    }

    return bestFitIndex;
  }

}
ComponentStateColorPalette.defaultPaletteConfig = {
  baseColor: parseColorHexRGB("#808080"),
  steps: 94
};

/**
 * @internal
 */
function contrast(a, b) {
  const L1 = a.relativeLuminance > b.relativeLuminance ? a : b;
  const L2 = a.relativeLuminance > b.relativeLuminance ? b : a;
  return (L1.relativeLuminance + 0.05) / (L2.relativeLuminance + 0.05);
}

/** @public */

const SwatchRGB = Object.freeze({
  create(r, g, b) {
    return new SwatchRGBImpl(r, g, b);
  },

  from(obj) {
    return new SwatchRGBImpl(obj.r, obj.g, obj.b);
  }

});
/**
 * Runtime test for an objects conformance with the SwatchRGB interface.
 * @internal
 */

function isSwatchRGB(value) {
  const test = {
    r: 0,
    g: 0,
    b: 0,
    toColorString: () => "",
    contrast: () => 0,
    relativeLuminance: 0
  };

  for (const key in test) {
    if (typeof test[key] !== typeof value[key]) {
      return false;
    }
  }

  return true;
}
/**
 * A RGB implementation of {@link Swatch}
 * @internal
 */

class SwatchRGBImpl extends ColorRGBA64 {
  /**
   *
   * @param red - Red channel expressed as a number between 0 and 1
   * @param green - Green channel expressed as a number between 0 and 1
   * @param blue - Blue channel expressed as a number between 0 and 1
   */
  constructor(red, green, blue) {
    super(red, green, blue, 1);
    this.toColorString = this.toStringHexRGB;
    this.contrast = contrast.bind(null, this);
    this.createCSS = this.toColorString;
    this.relativeLuminance = rgbToRelativeLuminance(this);
  }

  static fromObject(obj) {
    return new SwatchRGBImpl(obj.r, obj.g, obj.b);
  }

}

/**
 * @internal
 */
function binarySearch(valuesToSearch, searchCondition, startIndex = 0, endIndex = valuesToSearch.length - 1) {
  if (endIndex === startIndex) {
    return valuesToSearch[startIndex];
  }

  const middleIndex = Math.floor((endIndex - startIndex) / 2) + startIndex; // Check to see if this passes on the item in the center of the array
  // if it does check the previous values

  return searchCondition(valuesToSearch[middleIndex]) ? binarySearch(valuesToSearch, searchCondition, startIndex, middleIndex // include this index because it passed the search condition
  ) : binarySearch(valuesToSearch, searchCondition, middleIndex + 1, // exclude this index because it failed the search condition
  endIndex);
}

/*
 * A color is in "dark" if there is more contrast between #000000 and a reference
 * color than #FFFFFF and the reference color. That threshold can be expressed as a relative luminance
 * using the contrast formula as (1 + 0.5) / (R + 0.05) === (R + 0.05) / (0 + 0.05),
 * which reduces to the following, where 'R' is the relative luminance of the reference color
 */
const target = (-0.1 + Math.sqrt(0.21)) / 2;
/**
 * Determines if a color should be considered Dark Mode
 * @param color - The color to check to mode of
 * @returns boolean
 *
 * @public
 */

function isDark(color) {
  return color.relativeLuminance <= target;
}

/**
 * @internal
 */

function directionByIsDark(color) {
  return isDark(color) ? -1 : 1;
}

function create$1(rOrSource, g, b) {
  if (typeof rOrSource === "number") {
    /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */
    return PaletteRGB.from(SwatchRGB.create(rOrSource, g, b));
  } else {
    return PaletteRGB.from(rOrSource);
  }
}

function from(source) {
  return isSwatchRGB(source) ? PaletteRGBImpl.from(source) : PaletteRGBImpl.from(SwatchRGB.create(source.r, source.g, source.b));
}
/** @public */


const PaletteRGB = Object.freeze({
  create: create$1,
  from
});
/**
 * A {@link Palette} representing RGB swatch values.
 * @public
 */

class PaletteRGBImpl {
  /**
   *
   * @param source - The source color for the palette
   * @param swatches - All swatches in the palette
   */
  constructor(source, swatches) {
    this.closestIndexCache = new Map();
    this.source = source;
    this.swatches = swatches;
    this.reversedSwatches = Object.freeze([...this.swatches].reverse());
    this.lastIndex = this.swatches.length - 1;
  }
  /**
   * {@inheritdoc Palette.colorContrast}
   */


  colorContrast(reference, contrastTarget, initialSearchIndex, direction) {
    if (initialSearchIndex === undefined) {
      initialSearchIndex = this.closestIndexOf(reference);
    }

    let source = this.swatches;
    const endSearchIndex = this.lastIndex;
    let startSearchIndex = initialSearchIndex;

    if (direction === undefined) {
      direction = directionByIsDark(reference);
    }

    const condition = value => contrast(reference, value) >= contrastTarget;

    if (direction === -1) {
      source = this.reversedSwatches;
      startSearchIndex = endSearchIndex - startSearchIndex;
    }

    return binarySearch(source, condition, startSearchIndex, endSearchIndex);
  }
  /**
   * {@inheritdoc Palette.get}
   */


  get(index) {
    return this.swatches[index] || this.swatches[clamp(index, 0, this.lastIndex)];
  }
  /**
   * {@inheritdoc Palette.closestIndexOf}
   */


  closestIndexOf(reference) {
    if (this.closestIndexCache.has(reference.relativeLuminance)) {
      /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */
      return this.closestIndexCache.get(reference.relativeLuminance);
    }

    let index = this.swatches.indexOf(reference);

    if (index !== -1) {
      this.closestIndexCache.set(reference.relativeLuminance, index);
      return index;
    }

    const closest = this.swatches.reduce((previous, next) => Math.abs(next.relativeLuminance - reference.relativeLuminance) < Math.abs(previous.relativeLuminance - reference.relativeLuminance) ? next : previous);
    index = this.swatches.indexOf(closest);
    this.closestIndexCache.set(reference.relativeLuminance, index);
    return index;
  }
  /**
   * Create a color palette from a provided swatch
   * @param source - The source swatch to create a palette from
   * @returns
   */


  static from(source) {
    return new PaletteRGBImpl(source, Object.freeze(new ComponentStateColorPalette({
      /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */
      baseColor: ColorRGBA64.fromObject(source)
    }).palette.map(x => {
      /* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */
      const _x = parseColorHexRGB(x.toStringHexRGB());

      return SwatchRGB.create(_x.r, _x.g, _x.b);
    })));
  }

}

/**
 * @internal
 */
function accentFill(palette, neutralPalette, reference, hoverDelta, activeDelta, focusDelta, neutralFillRestDelta, neutralFillHoverDelta, neutralFillActiveDelta) {
  const accent = palette.source;
  const referenceIndex = neutralPalette.closestIndexOf(reference);
  const swapThreshold = Math.max(neutralFillRestDelta, neutralFillHoverDelta, neutralFillActiveDelta);
  const direction = referenceIndex >= swapThreshold ? -1 : 1;
  const accentIndex = palette.closestIndexOf(accent);
  const hoverIndex = accentIndex;
  const restIndex = hoverIndex + direction * -1 * hoverDelta;
  const activeIndex = restIndex + direction * activeDelta;
  const focusIndex = restIndex + direction * focusDelta;
  return {
    rest: palette.get(restIndex),
    hover: palette.get(hoverIndex),
    active: palette.get(activeIndex),
    focus: palette.get(focusIndex)
  };
}

/**
 * @internal
 */

function accentForeground(palette, reference, contrastTarget, restDelta, hoverDelta, activeDelta, focusDelta) {
  const accent = palette.source;
  const accentIndex = palette.closestIndexOf(accent);
  const direction = directionByIsDark(reference);
  const startIndex = accentIndex + (direction === 1 ? Math.min(restDelta, hoverDelta) : Math.max(direction * restDelta, direction * hoverDelta));
  const accessibleSwatch = palette.colorContrast(reference, contrastTarget, startIndex, direction);
  const accessibleIndex1 = palette.closestIndexOf(accessibleSwatch);
  const accessibleIndex2 = accessibleIndex1 + direction * Math.abs(restDelta - hoverDelta);
  const indexOneIsRestState = direction === 1 ? restDelta < hoverDelta : direction * restDelta > direction * hoverDelta;
  let restIndex;
  let hoverIndex;

  if (indexOneIsRestState) {
    restIndex = accessibleIndex1;
    hoverIndex = accessibleIndex2;
  } else {
    restIndex = accessibleIndex2;
    hoverIndex = accessibleIndex1;
  }

  return {
    rest: palette.get(restIndex),
    hover: palette.get(hoverIndex),
    active: palette.get(restIndex + direction * activeDelta),
    focus: palette.get(restIndex + direction * focusDelta)
  };
}

/**
 * @internal
 */

const white = SwatchRGB.create(1, 1, 1);
/**
 * @internal
 */

const black = SwatchRGB.create(0, 0, 0);
/**
 * @internal
 */

/* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */

const middleGrey = SwatchRGB.from(parseColorHexRGB("#808080"));
/**
 * @internal
 */

/* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */

const accentBase = SwatchRGB.from(parseColorHexRGB("#DA1A5F"));

/**
 * @internal
 */

function foregroundOnAccent(reference, contrastTarget) {
  return reference.contrast(white) >= contrastTarget ? white : black;
}

/**
 *
 * @param palette - The palette to operate on
 * @param reference - The reference color to calculate a color for
 * @param delta - The offset from the reference's location
 * @param threshold - Determines if a lighter or darker color than the reference will be picked.
 * @returns
 *
 * @internal
 */
function neutralFill(palette, reference, restDelta, hoverDelta, activeDelta, focusDelta) {
  const referenceIndex = palette.closestIndexOf(reference);
  const threshold = Math.max(restDelta, hoverDelta, activeDelta, focusDelta);
  const direction = referenceIndex >= threshold ? -1 : 1;
  return {
    rest: palette.get(referenceIndex + direction * restDelta),
    hover: palette.get(referenceIndex + direction * hoverDelta),
    active: palette.get(referenceIndex + direction * activeDelta),
    focus: palette.get(referenceIndex + direction * focusDelta)
  };
}

/**
 * @internal
 */

function neutralFillInput(palette, reference, restDelta, hoverDelta, activeDelta, focusDelta) {
  const direction = directionByIsDark(reference);
  const referenceIndex = palette.closestIndexOf(reference);
  return {
    rest: palette.get(referenceIndex - direction * restDelta),
    hover: palette.get(referenceIndex - direction * hoverDelta),
    active: palette.get(referenceIndex - direction * activeDelta),
    focus: palette.get(referenceIndex - direction * focusDelta)
  };
}

/**
 * @internal
 */
function neutralFillLayer(palette, reference, delta) {
  const referenceIndex = palette.closestIndexOf(reference);
  return palette.get(referenceIndex - (referenceIndex < delta ? delta * -1 : delta));
}

/**
 * @internal
 */
function neutralFillStealth(palette, reference, restDelta, hoverDelta, activeDelta, focusDelta, fillRestDelta, fillHoverDelta, fillActiveDelta, fillFocusDelta) {
  const swapThreshold = Math.max(restDelta, hoverDelta, activeDelta, focusDelta, fillRestDelta, fillHoverDelta, fillActiveDelta, fillFocusDelta);
  const referenceIndex = palette.closestIndexOf(reference);
  const direction = referenceIndex >= swapThreshold ? -1 : 1;
  return {
    rest: palette.get(referenceIndex + direction * restDelta),
    hover: palette.get(referenceIndex + direction * hoverDelta),
    active: palette.get(referenceIndex + direction * activeDelta),
    focus: palette.get(referenceIndex + direction * focusDelta)
  };
}

/**
 * @internal
 */

function neutralFillContrast(palette, reference, restDelta, hoverDelta, activeDelta, focusDelta) {
  const direction = directionByIsDark(reference);
  const accessibleIndex = palette.closestIndexOf(palette.colorContrast(reference, 4.5));
  const accessibleIndex2 = accessibleIndex + direction * Math.abs(restDelta - hoverDelta);
  const indexOneIsRest = direction === 1 ? restDelta < hoverDelta : direction * restDelta > direction * hoverDelta;
  let restIndex;
  let hoverIndex;

  if (indexOneIsRest) {
    restIndex = accessibleIndex;
    hoverIndex = accessibleIndex2;
  } else {
    restIndex = accessibleIndex2;
    hoverIndex = accessibleIndex;
  }

  return {
    rest: palette.get(restIndex),
    hover: palette.get(hoverIndex),
    active: palette.get(restIndex + direction * activeDelta),
    focus: palette.get(restIndex + direction * focusDelta)
  };
}

/** @internal */

function focusStrokeOuter$1(palette, reference) {
  return palette.colorContrast(reference, 3.5);
}
/** @internal */

function focusStrokeInner$1(palette, reference, focusColor) {
  return palette.colorContrast(focusColor, 3.5, palette.closestIndexOf(palette.source), directionByIsDark(reference) * -1);
}

/**
 * @internal
 */
function neutralForeground(palette, reference) {
  return palette.colorContrast(reference, 14);
}

/**
 * The neutralForegroundHint color recipe
 * @param palette - The palette to operate on
 * @param reference - The reference color
 *
 * @internal
 */
function neutralForegroundHint$1(palette, reference) {
  return palette.colorContrast(reference, 4.5);
}

function baseLayerLuminanceSwatch(luminance) {
  return SwatchRGB.create(luminance, luminance, luminance);
}
/**
 * Recommended values for light and dark mode for {@link @microsoft/fast-components#baseLayerLuminance}.
 *
 * @public
 */

const StandardLuminance = {
  LightMode: 1,
  DarkMode: 0.23
};

/**
 * @internal
 */

function neutralLayerCardContainer$1(palette, relativeLuminance, layerDelta) {
  return palette.get(palette.closestIndexOf(baseLayerLuminanceSwatch(relativeLuminance)) + layerDelta);
}

/**
 * @internal
 */

function neutralLayerFloating$1(palette, relativeLuminance, layerDelta) {
  const cardIndex = palette.closestIndexOf(baseLayerLuminanceSwatch(relativeLuminance)) - layerDelta;
  return palette.get(cardIndex - layerDelta);
}

function neutralLayer1$1(palette, baseLayerLuminance) {
  return palette.get(palette.closestIndexOf(baseLayerLuminanceSwatch(baseLayerLuminance)));
}

/**
 * @internal
 */

function neutralLayer2Index(palette, luminance, layerDelta, fillRestDelta, fillHoverDelta, fillActiveDelta) {
  return Math.max(palette.closestIndexOf(baseLayerLuminanceSwatch(luminance)) + layerDelta, fillRestDelta, fillHoverDelta, fillActiveDelta);
}
/**
 * @internal
 */

function neutralLayer2$1(palette, luminance, layerDelta, fillRestDelta, fillHoverDelta, fillActiveDelta) {
  return palette.get(neutralLayer2Index(palette, luminance, layerDelta, fillRestDelta, fillHoverDelta, fillActiveDelta));
}

/**
 * @internal
 */

function neutralLayer3$1(palette, luminance, layerDelta, fillRestDelta, fillHoverDelta, fillActiveDelta) {
  return palette.get(neutralLayer2Index(palette, luminance, layerDelta, fillRestDelta, fillHoverDelta, fillActiveDelta) + layerDelta);
}

/**
 * @internal
 */

function neutralLayer4$1(palette, luminance, layerDelta, fillRestDelta, fillHoverDelta, fillActiveDelta) {
  return palette.get(neutralLayer2Index(palette, luminance, layerDelta, fillRestDelta, fillHoverDelta, fillActiveDelta) + layerDelta * 2);
}

/**
 * @internal
 */

function neutralStroke(palette, reference, restDelta, hoverDelta, activeDelta, focusDelta) {
  const referenceIndex = palette.closestIndexOf(reference);
  const direction = directionByIsDark(reference);
  const restIndex = referenceIndex + direction * restDelta;
  const hoverIndex = restIndex + direction * (hoverDelta - restDelta);
  const activeIndex = restIndex + direction * (activeDelta - restDelta);
  const focusIndex = restIndex + direction * (focusDelta - restDelta);
  return {
    rest: palette.get(restIndex),
    hover: palette.get(hoverIndex),
    active: palette.get(activeIndex),
    focus: palette.get(focusIndex)
  };
}

/**
 * The neutralStrokeDivider color recipe
 * @param palette - The palette to operate on
 * @param reference - The reference color
 * @param delta - The offset from the reference
 *
 * @internal
 */

function neutralStrokeDivider(palette, reference, delta) {
  return palette.get(palette.closestIndexOf(reference) + directionByIsDark(reference) * delta);
}

const {
  create
} = DesignToken;

function createNonCss(name) {
  return DesignToken.create({
    name,
    cssCustomPropertyName: null
  });
} // General tokens

/** @public */


const bodyFont = create("body-font").withDefault('aktiv-grotesk, "Segoe UI", Arial, Helvetica, sans-serif');
/** @public */

const baseHeightMultiplier = create("base-height-multiplier").withDefault(10);
/** @public */

const baseHorizontalSpacingMultiplier = create("base-horizontal-spacing-multiplier").withDefault(3);
/** @public */

const baseLayerLuminance = create("base-layer-luminance").withDefault(StandardLuminance.DarkMode);
/** @public */

const controlCornerRadius = create("control-corner-radius").withDefault(4);
/** @public */

const density = create("density").withDefault(0);
/** @public */

const designUnit = create("design-unit").withDefault(4);
/** @public */

const direction = create("direction").withDefault(Direction.ltr);
/** @public */

const disabledOpacity = create("disabled-opacity").withDefault(0.3);
/** @public */

const strokeWidth = create("stroke-width").withDefault(1);
/** @public */

const focusStrokeWidth = create("focus-stroke-width").withDefault(2); // Typography values

/** @public */

const typeRampBaseFontSize = create("type-ramp-base-font-size").withDefault("14px");
/** @public */

const typeRampBaseLineHeight = create("type-ramp-base-line-height").withDefault("20px");
/** @public */

const typeRampMinus1FontSize = create("type-ramp-minus-1-font-size").withDefault("12px");
/** @public */

const typeRampMinus1LineHeight = create("type-ramp-minus-1-line-height").withDefault("16px");
/** @public */

const typeRampMinus2FontSize = create("type-ramp-minus-2-font-size").withDefault("10px");
/** @public */

const typeRampMinus2LineHeight = create("type-ramp-minus-2-line-height").withDefault("16px");
/** @public */

const typeRampPlus1FontSize = create("type-ramp-plus-1-font-size").withDefault("16px");
/** @public */

const typeRampPlus1LineHeight = create("type-ramp-plus-1-line-height").withDefault("24px");
/** @public */

const typeRampPlus2FontSize = create("type-ramp-plus-2-font-size").withDefault("20px");
/** @public */

const typeRampPlus2LineHeight = create("type-ramp-plus-2-line-height").withDefault("28px");
/** @public */

const typeRampPlus3FontSize = create("type-ramp-plus-3-font-size").withDefault("28px");
/** @public */

const typeRampPlus3LineHeight = create("type-ramp-plus-3-line-height").withDefault("36px");
/** @public */

const typeRampPlus4FontSize = create("type-ramp-plus-4-font-size").withDefault("34px");
/** @public */

const typeRampPlus4LineHeight = create("type-ramp-plus-4-line-height").withDefault("44px");
/** @public */

const typeRampPlus5FontSize = create("type-ramp-plus-5-font-size").withDefault("46px");
/** @public */

const typeRampPlus5LineHeight = create("type-ramp-plus-5-line-height").withDefault("56px");
/** @public */

const typeRampPlus6FontSize = create("type-ramp-plus-6-font-size").withDefault("60px");
/** @public */

const typeRampPlus6LineHeight = create("type-ramp-plus-6-line-height").withDefault("72px"); // Color recipe values

/** @public */

const accentFillRestDelta = createNonCss("accent-fill-rest-delta").withDefault(0);
/** @public */

const accentFillHoverDelta = createNonCss("accent-fill-hover-delta").withDefault(4);
/** @public */

const accentFillActiveDelta = createNonCss("accent-fill-active-delta").withDefault(-5);
/** @public */

const accentFillFocusDelta = createNonCss("accent-fill-focus-delta").withDefault(0);
/** @public */

const accentForegroundRestDelta = createNonCss("accent-foreground-rest-delta").withDefault(0);
/** @public */

const accentForegroundHoverDelta = createNonCss("accent-foreground-hover-delta").withDefault(6);
/** @public */

const accentForegroundActiveDelta = createNonCss("accent-foreground-active-delta").withDefault(-4);
/** @public */

const accentForegroundFocusDelta = createNonCss("accent-foreground-focus-delta").withDefault(0);
/** @public */

const neutralFillRestDelta = createNonCss("neutral-fill-rest-delta").withDefault(7);
/** @public */

const neutralFillHoverDelta = createNonCss("neutral-fill-hover-delta").withDefault(10);
/** @public */

const neutralFillActiveDelta = createNonCss("neutral-fill-active-delta").withDefault(5);
/** @public */

const neutralFillFocusDelta = createNonCss("neutral-fill-focus-delta").withDefault(0);
/** @public */

const neutralFillInputRestDelta = createNonCss("neutral-fill-input-rest-delta").withDefault(0);
/** @public */

const neutralFillInputHoverDelta = createNonCss("neutral-fill-input-hover-delta").withDefault(0);
/** @public */

const neutralFillInputActiveDelta = createNonCss("neutral-fill-input-active-delta").withDefault(0);
/** @public */

const neutralFillInputFocusDelta = createNonCss("neutral-fill-input-focus-delta").withDefault(0);
/** @public */

const neutralFillStealthRestDelta = createNonCss("neutral-fill-stealth-rest-delta").withDefault(0);
/** @public */

const neutralFillStealthHoverDelta = createNonCss("neutral-fill-stealth-hover-delta").withDefault(5);
/** @public */

const neutralFillStealthActiveDelta = createNonCss("neutral-fill-stealth-active-delta").withDefault(3);
/** @public */

const neutralFillStealthFocusDelta = createNonCss("neutral-fill-stealth-focus-delta").withDefault(0);
/** @public */

const neutralFillStrongRestDelta = createNonCss("neutral-fill-strong-rest-delta").withDefault(0);
/** @public */

const neutralFillStrongHoverDelta = createNonCss("neutral-fill-strong-hover-delta").withDefault(8);
/** @public */

const neutralFillStrongActiveDelta = createNonCss("neutral-fill-strong-active-delta").withDefault(-5);
/** @public */

const neutralFillStrongFocusDelta = createNonCss("neutral-fill-strong-focus-delta").withDefault(0);
/** @public */

const neutralFillLayerRestDelta = createNonCss("neutral-fill-layer-rest-delta").withDefault(3);
/** @public */

const neutralStrokeRestDelta = createNonCss("neutral-stroke-rest-delta").withDefault(25);
/** @public */

const neutralStrokeHoverDelta = createNonCss("neutral-stroke-hover-delta").withDefault(40);
/** @public */

const neutralStrokeActiveDelta = createNonCss("neutral-stroke-active-delta").withDefault(16);
/** @public */

const neutralStrokeFocusDelta = createNonCss("neutral-stroke-focus-delta").withDefault(25);
/** @public */

const neutralStrokeDividerRestDelta = createNonCss("neutral-stroke-divider-rest-delta").withDefault(8); // Color recipes

/** @public */

const neutralColor = create("neutral-color").withDefault(middleGrey);
/** @public */

const neutralPalette = createNonCss("neutral-palette").withDefault(element => PaletteRGB.from(neutralColor.getValueFor(element)));
/** @public */

const accentColor = create("accent-color").withDefault(accentBase);
/** @public */

const accentPalette = createNonCss("accent-palette").withDefault(element => PaletteRGB.from(accentColor.getValueFor(element))); // Neutral Layer Card Container

/** @public */

const neutralLayerCardContainerRecipe = createNonCss("neutral-layer-card-container-recipe").withDefault({
  evaluate: element => neutralLayerCardContainer$1(neutralPalette.getValueFor(element), baseLayerLuminance.getValueFor(element), neutralFillLayerRestDelta.getValueFor(element))
});
/** @public */

const neutralLayerCardContainer = create("neutral-layer-card-container").withDefault(element => neutralLayerCardContainerRecipe.getValueFor(element).evaluate(element)); // Neutral Layer Floating

/** @public */

const neutralLayerFloatingRecipe = createNonCss("neutral-layer-floating-recipe").withDefault({
  evaluate: element => neutralLayerFloating$1(neutralPalette.getValueFor(element), baseLayerLuminance.getValueFor(element), neutralFillLayerRestDelta.getValueFor(element))
});
/** @public */

const neutralLayerFloating = create("neutral-layer-floating").withDefault(element => neutralLayerFloatingRecipe.getValueFor(element).evaluate(element)); // Neutral Layer 1

/** @public */

const neutralLayer1Recipe = createNonCss("neutral-layer-1-recipe").withDefault({
  evaluate: element => neutralLayer1$1(neutralPalette.getValueFor(element), baseLayerLuminance.getValueFor(element))
});
/** @public */

const neutralLayer1 = create("neutral-layer-1").withDefault(element => neutralLayer1Recipe.getValueFor(element).evaluate(element)); // Neutral Layer 2

/** @public */

const neutralLayer2Recipe = createNonCss("neutral-layer-2-recipe").withDefault({
  evaluate: element => neutralLayer2$1(neutralPalette.getValueFor(element), baseLayerLuminance.getValueFor(element), neutralFillLayerRestDelta.getValueFor(element), neutralFillRestDelta.getValueFor(element), neutralFillHoverDelta.getValueFor(element), neutralFillActiveDelta.getValueFor(element))
});
/** @public */

const neutralLayer2 = create("neutral-layer-2").withDefault(element => neutralLayer2Recipe.getValueFor(element).evaluate(element)); // Neutral Layer 3

/** @public */

const neutralLayer3Recipe = createNonCss("neutral-layer-3-recipe").withDefault({
  evaluate: element => neutralLayer3$1(neutralPalette.getValueFor(element), baseLayerLuminance.getValueFor(element), neutralFillLayerRestDelta.getValueFor(element), neutralFillRestDelta.getValueFor(element), neutralFillHoverDelta.getValueFor(element), neutralFillActiveDelta.getValueFor(element))
});
/** @public */

const neutralLayer3 = create("neutral-layer-3").withDefault(element => neutralLayer3Recipe.getValueFor(element).evaluate(element)); // Neutral Layer 4

/** @public */

const neutralLayer4Recipe = createNonCss("neutral-layer-4-recipe").withDefault({
  evaluate: element => neutralLayer4$1(neutralPalette.getValueFor(element), baseLayerLuminance.getValueFor(element), neutralFillLayerRestDelta.getValueFor(element), neutralFillRestDelta.getValueFor(element), neutralFillHoverDelta.getValueFor(element), neutralFillActiveDelta.getValueFor(element))
});
/** @public */

const neutralLayer4 = create("neutral-layer-4").withDefault(element => neutralLayer4Recipe.getValueFor(element).evaluate(element));
/** @public */

const fillColor = create("fill-color").withDefault(element => neutralLayer1.getValueFor(element));
var ContrastTarget;

(function (ContrastTarget) {
  ContrastTarget[ContrastTarget["normal"] = 4.5] = "normal";
  ContrastTarget[ContrastTarget["large"] = 7] = "large";
})(ContrastTarget || (ContrastTarget = {})); // Accent Fill

/** @public */


const accentFillRecipe = create({
  name: "accent-fill-recipe",
  cssCustomPropertyName: null
}).withDefault({
  evaluate: (element, reference) => accentFill(accentPalette.getValueFor(element), neutralPalette.getValueFor(element), reference || fillColor.getValueFor(element), accentFillHoverDelta.getValueFor(element), accentFillActiveDelta.getValueFor(element), accentFillFocusDelta.getValueFor(element), neutralFillRestDelta.getValueFor(element), neutralFillHoverDelta.getValueFor(element), neutralFillActiveDelta.getValueFor(element))
});
/** @public */

const accentFillRest = create("accent-fill-rest").withDefault(element => {
  return accentFillRecipe.getValueFor(element).evaluate(element).rest;
});
/** @public */

const accentFillHover = create("accent-fill-hover").withDefault(element => {
  return accentFillRecipe.getValueFor(element).evaluate(element).hover;
});
/** @public */

const accentFillActive = create("accent-fill-active").withDefault(element => {
  return accentFillRecipe.getValueFor(element).evaluate(element).active;
});
/** @public */

const accentFillFocus = create("accent-fill-focus").withDefault(element => {
  return accentFillRecipe.getValueFor(element).evaluate(element).focus;
}); // Foreground On Accent

const foregroundOnAccentByContrast = contrast => (element, reference) => {
  return foregroundOnAccent(reference || accentFillRest.getValueFor(element), contrast);
};
/** @public */


const foregroundOnAccentRecipe = createNonCss("foreground-on-accent-recipe").withDefault({
  evaluate: (element, reference) => foregroundOnAccentByContrast(ContrastTarget.normal)(element, reference)
});
/** @public */

const foregroundOnAccentRest = create("foreground-on-accent-rest").withDefault(element => foregroundOnAccentRecipe.getValueFor(element).evaluate(element, accentFillRest.getValueFor(element)));
/** @public */

const foregroundOnAccentHover = create("foreground-on-accent-hover").withDefault(element => foregroundOnAccentRecipe.getValueFor(element).evaluate(element, accentFillHover.getValueFor(element)));
/** @public */

const foregroundOnAccentActive = create("foreground-on-accent-active").withDefault(element => foregroundOnAccentRecipe.getValueFor(element).evaluate(element, accentFillActive.getValueFor(element)));
/** @public */

const foregroundOnAccentFocus = create("foreground-on-accent-focus").withDefault(element => foregroundOnAccentRecipe.getValueFor(element).evaluate(element, accentFillFocus.getValueFor(element)));
/** @public */

const foregroundOnAccentLargeRecipe = createNonCss("foreground-on-accent-large-recipe").withDefault({
  evaluate: (element, reference) => foregroundOnAccentByContrast(ContrastTarget.large)(element, reference)
});
/** @public */

const foregroundOnAccentRestLarge = create("foreground-on-accent-rest-large").withDefault(element => foregroundOnAccentLargeRecipe.getValueFor(element).evaluate(element, accentFillRest.getValueFor(element)));
/** @public */

const foregroundOnAccentHoverLarge = create("foreground-on-accent-hover-large").withDefault(element => foregroundOnAccentLargeRecipe.getValueFor(element).evaluate(element, accentFillHover.getValueFor(element)));
/** @public */

const foregroundOnAccentActiveLarge = create("foreground-on-accent-active-large").withDefault(element => foregroundOnAccentLargeRecipe.getValueFor(element).evaluate(element, accentFillActive.getValueFor(element)));
/** @public */

const foregroundOnAccentFocusLarge = create("foreground-on-accent-focus-large").withDefault(element => foregroundOnAccentLargeRecipe.getValueFor(element).evaluate(element, accentFillFocus.getValueFor(element))); // Accent Foreground

const accentForegroundByContrast = contrast => (element, reference) => accentForeground(accentPalette.getValueFor(element), reference || fillColor.getValueFor(element), contrast, accentForegroundRestDelta.getValueFor(element), accentForegroundHoverDelta.getValueFor(element), accentForegroundActiveDelta.getValueFor(element), accentForegroundFocusDelta.getValueFor(element));
/** @public */


const accentForegroundRecipe = create({
  name: "accent-foreground-recipe",
  cssCustomPropertyName: null
}).withDefault({
  evaluate: (element, reference) => accentForegroundByContrast(ContrastTarget.normal)(element, reference)
});
/** @public */

const accentForegroundRest = create("accent-foreground-rest").withDefault(element => accentForegroundRecipe.getValueFor(element).evaluate(element).rest);
/** @public */

const accentForegroundHover = create("accent-foreground-hover").withDefault(element => accentForegroundRecipe.getValueFor(element).evaluate(element).hover);
/** @public */

const accentForegroundActive = create("accent-foreground-active").withDefault(element => accentForegroundRecipe.getValueFor(element).evaluate(element).active);
/** @public */

const accentForegroundFocus = create("accent-foreground-focus").withDefault(element => accentForegroundRecipe.getValueFor(element).evaluate(element).focus); // Neutral Fill

/** @public */

const neutralFillRecipe = create({
  name: "neutral-fill-recipe",
  cssCustomPropertyName: null
}).withDefault({
  evaluate: (element, reference) => neutralFill(neutralPalette.getValueFor(element), reference || fillColor.getValueFor(element), neutralFillRestDelta.getValueFor(element), neutralFillHoverDelta.getValueFor(element), neutralFillActiveDelta.getValueFor(element), neutralFillFocusDelta.getValueFor(element))
});
/** @public */

const neutralFillRest = create("neutral-fill-rest").withDefault(element => neutralFillRecipe.getValueFor(element).evaluate(element).rest);
/** @public */

const neutralFillHover = create("neutral-fill-hover").withDefault(element => neutralFillRecipe.getValueFor(element).evaluate(element).hover);
/** @public */

const neutralFillActive = create("neutral-fill-active").withDefault(element => neutralFillRecipe.getValueFor(element).evaluate(element).active);
/** @public */

const neutralFillFocus = create("neutral-fill-focus").withDefault(element => neutralFillRecipe.getValueFor(element).evaluate(element).focus); // Neutral Fill Input

/** @public */

const neutralFillInputRecipe = create({
  name: "neutral-fill-input-recipe",
  cssCustomPropertyName: null
}).withDefault({
  evaluate: (element, reference) => neutralFillInput(neutralPalette.getValueFor(element), reference || fillColor.getValueFor(element), neutralFillInputRestDelta.getValueFor(element), neutralFillInputHoverDelta.getValueFor(element), neutralFillInputActiveDelta.getValueFor(element), neutralFillInputFocusDelta.getValueFor(element))
});
/** @public */

const neutralFillInputRest = create("neutral-fill-input-rest").withDefault(element => neutralFillInputRecipe.getValueFor(element).evaluate(element).rest);
/** @public */

const neutralFillInputHover = create("neutral-fill-input-hover").withDefault(element => neutralFillInputRecipe.getValueFor(element).evaluate(element).hover);
/** @public */

const neutralFillInputActive = create("neutral-fill-input-active").withDefault(element => neutralFillInputRecipe.getValueFor(element).evaluate(element).active);
/** @public */

const neutralFillInputFocus = create("neutral-fill-input-focus").withDefault(element => neutralFillInputRecipe.getValueFor(element).evaluate(element).focus); // Neutral Fill Stealth

/** @public */

const neutralFillStealthRecipe = create({
  name: "neutral-fill-stealth-recipe",
  cssCustomPropertyName: null
}).withDefault({
  evaluate: (element, reference) => neutralFillStealth(neutralPalette.getValueFor(element), reference || fillColor.getValueFor(element), neutralFillStealthRestDelta.getValueFor(element), neutralFillStealthHoverDelta.getValueFor(element), neutralFillStealthActiveDelta.getValueFor(element), neutralFillStealthFocusDelta.getValueFor(element), neutralFillRestDelta.getValueFor(element), neutralFillHoverDelta.getValueFor(element), neutralFillActiveDelta.getValueFor(element), neutralFillFocusDelta.getValueFor(element))
});
/** @public */

const neutralFillStealthRest = create("neutral-fill-stealth-rest").withDefault(element => neutralFillStealthRecipe.getValueFor(element).evaluate(element).rest);
/** @public */

const neutralFillStealthHover = create("neutral-fill-stealth-hover").withDefault(element => neutralFillStealthRecipe.getValueFor(element).evaluate(element).hover);
/** @public */

const neutralFillStealthActive = create("neutral-fill-stealth-active").withDefault(element => neutralFillStealthRecipe.getValueFor(element).evaluate(element).active);
/** @public */

const neutralFillStealthFocus = create("neutral-fill-stealth-focus").withDefault(element => neutralFillStealthRecipe.getValueFor(element).evaluate(element).focus); // Neutral Fill Strong

/** @public */

const neutralFillStrongRecipe = create({
  name: "neutral-fill-strong-recipe",
  cssCustomPropertyName: null
}).withDefault({
  evaluate: (element, reference) => neutralFillContrast(neutralPalette.getValueFor(element), reference || fillColor.getValueFor(element), neutralFillStrongRestDelta.getValueFor(element), neutralFillStrongHoverDelta.getValueFor(element), neutralFillStrongActiveDelta.getValueFor(element), neutralFillStrongFocusDelta.getValueFor(element))
});
/** @public */

const neutralFillStrongRest = create("neutral-fill-strong-rest").withDefault(element => neutralFillStrongRecipe.getValueFor(element).evaluate(element).rest);
/** @public */

const neutralFillStrongHover = create("neutral-fill-strong-hover").withDefault(element => neutralFillStrongRecipe.getValueFor(element).evaluate(element).hover);
/** @public */

const neutralFillStrongActive = create("neutral-fill-strong-active").withDefault(element => neutralFillStrongRecipe.getValueFor(element).evaluate(element).active);
/** @public */

const neutralFillStrongFocus = create("neutral-fill-strong-focus").withDefault(element => neutralFillStrongRecipe.getValueFor(element).evaluate(element).focus); // Neutral Fill Layer

/** @public */

const neutralFillLayerRecipe = createNonCss("neutral-fill-layer-recipe").withDefault({
  evaluate: (element, reference) => neutralFillLayer(neutralPalette.getValueFor(element), reference || fillColor.getValueFor(element), neutralFillLayerRestDelta.getValueFor(element))
});
/** @public */

const neutralFillLayerRest = create("neutral-fill-layer-rest").withDefault(element => neutralFillLayerRecipe.getValueFor(element).evaluate(element)); // Focus Stroke Outer

/** @public */

const focusStrokeOuterRecipe = createNonCss("focus-stroke-outer-recipe").withDefault({
  evaluate: element => focusStrokeOuter$1(neutralPalette.getValueFor(element), fillColor.getValueFor(element))
});
/** @public */

const focusStrokeOuter = create("focus-stroke-outer").withDefault(element => focusStrokeOuterRecipe.getValueFor(element).evaluate(element)); // Focus Stroke Inner

/** @public */

const focusStrokeInnerRecipe = createNonCss("focus-stroke-inner-recipe").withDefault({
  evaluate: element => focusStrokeInner$1(accentPalette.getValueFor(element), fillColor.getValueFor(element), focusStrokeOuter.getValueFor(element))
});
/** @public */

const focusStrokeInner = create("focus-stroke-inner").withDefault(element => focusStrokeInnerRecipe.getValueFor(element).evaluate(element)); // Neutral Foreground Hint

/** @public */

const neutralForegroundHintRecipe = createNonCss("neutral-foreground-hint-recipe").withDefault({
  evaluate: element => neutralForegroundHint$1(neutralPalette.getValueFor(element), fillColor.getValueFor(element))
});
/** @public */

const neutralForegroundHint = create("neutral-foreground-hint").withDefault(element => neutralForegroundHintRecipe.getValueFor(element).evaluate(element)); // Neutral Foreground

/** @public */

const neutralForegroundRecipe = createNonCss("neutral-foreground-recipe").withDefault({
  evaluate: element => neutralForeground(neutralPalette.getValueFor(element), fillColor.getValueFor(element))
});
/** @public */

const neutralForegroundRest = create("neutral-foreground-rest").withDefault(element => neutralForegroundRecipe.getValueFor(element).evaluate(element)); // Neutral Stroke

/** @public */

const neutralStrokeRecipe = create({
  name: "neutral-stroke-recipe",
  cssCustomPropertyName: null
}).withDefault({
  evaluate: element => {
    return neutralStroke(neutralPalette.getValueFor(element), fillColor.getValueFor(element), neutralStrokeRestDelta.getValueFor(element), neutralStrokeHoverDelta.getValueFor(element), neutralStrokeActiveDelta.getValueFor(element), neutralStrokeFocusDelta.getValueFor(element));
  }
});
/** @public */

const neutralStrokeRest = create("neutral-stroke-rest").withDefault(element => neutralStrokeRecipe.getValueFor(element).evaluate(element).rest);
/** @public */

const neutralStrokeHover = create("neutral-stroke-hover").withDefault(element => neutralStrokeRecipe.getValueFor(element).evaluate(element).hover);
/** @public */

const neutralStrokeActive = create("neutral-stroke-active").withDefault(element => neutralStrokeRecipe.getValueFor(element).evaluate(element).active);
/** @public */

const neutralStrokeFocus = create("neutral-stroke-focus").withDefault(element => neutralStrokeRecipe.getValueFor(element).evaluate(element).focus); // Neutral Stroke Divider

/** @public */

const neutralStrokeDividerRecipe = createNonCss("neutral-stroke-divider-recipe").withDefault({
  evaluate: (element, reference) => neutralStrokeDivider(neutralPalette.getValueFor(element), reference || fillColor.getValueFor(element), neutralStrokeDividerRestDelta.getValueFor(element))
});
/** @public */

const neutralStrokeDividerRest = create("neutral-stroke-divider-rest").withDefault(element => neutralStrokeDividerRecipe.getValueFor(element).evaluate(element));
/**
 * The control height formula expressed as a design token.
 * This token does not provide a CSS custom property.
 *
 * @public
 */

const heightNumberAsToken = DesignToken.create({
  name: "height-number",
  cssCustomPropertyName: null
}).withDefault(target => (baseHeightMultiplier.getValueFor(target) + density.getValueFor(target)) * designUnit.getValueFor(target));

/**
 * Styles for Accordion
 * @public
 */

const accordionStyles = (context, definition) => css`
        ${display("flex")} :host{box-sizing:border-box;flex-direction:column;font-family:${bodyFont};font-size:${typeRampMinus1FontSize};line-height:${typeRampMinus1LineHeight};color:${neutralForegroundRest};border-top:calc(${strokeWidth} * 1px) solid ${neutralStrokeDividerRest}}`;

/**
 * A formula to retrieve the control height.
 * Use this as the value of any CSS property that
 * accepts a pixel size.
 */

const heightNumber = cssPartial`(${baseHeightMultiplier} + ${density}) * ${designUnit}`;

/**
 * Styles for AccordionItem
 * @public
 */

const accordionItemStyles = (context, definition) => css`
        ${display("flex")} :host{box-sizing:border-box;font-family:${bodyFont};flex-direction:column;font-size:${typeRampMinus1FontSize};line-height:${typeRampMinus1LineHeight};border-bottom:calc(${strokeWidth} * 1px) solid ${neutralStrokeDividerRest}}.region{display:none;padding:calc((6 + (${designUnit} * 2 * ${density})) * 1px)}.heading{display:grid;position:relative;grid-template-columns:auto 1fr auto calc(${heightNumber} * 1px)}.button{appearance:none;border:none;background:none;grid-column:2;grid-row:1;outline:none;padding:0 calc((6 + (${designUnit} * 2 * ${density})) * 1px);text-align:left;height:calc(${heightNumber} * 1px);color:${neutralForegroundRest};cursor:pointer;font-family:inherit}.button:hover{color:${neutralForegroundRest}}.button:active{color:${neutralForegroundRest}}.button::before{content:"";position:absolute;top:0;left:0;right:0;bottom:0;cursor:pointer}.button:${focusVisible}::before{outline:none;border:calc(${focusStrokeWidth} * 1px) solid ${focusStrokeOuter};border-radius:calc(${controlCornerRadius} * 1px)}:host([expanded]) .region{display:block}.icon{display:flex;align-items:center;justify-content:center;grid-column:4;pointer-events:none;position:relative}slot[name="expanded-icon"],slot[name="collapsed-icon"]{fill:${accentForegroundRest}}slot[name="collapsed-icon"]{display:flex}:host([expanded]) slot[name="collapsed-icon"]{display:none}slot[name="expanded-icon"]{display:none}:host([expanded]) slot[name="expanded-icon"]{display:flex}.start{display:flex;align-items:center;padding-inline-start:calc(${designUnit} * 1px);justify-content:center;grid-column:1;position:relative}.end{display:flex;align-items:center;justify-content:center;grid-column:3;position:relative}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            .button:${focusVisible}::before{border-color:${SystemColors.Highlight}}:host slot[name="collapsed-icon"],:host([expanded]) slot[name="expanded-icon"]{fill:${SystemColors.ButtonText}}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#AccordionItem} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#accordionItemTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-accordion-item>`
 */

const fastAccordionItem = AccordionItem.compose({
  baseName: "accordion-item",
  template: accordionItemTemplate,
  styles: accordionItemStyles,
  collapsedIcon:
  /* html */
  `
        <svg
            width="20"
            height="20"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                fill-rule="evenodd"
                clip-rule="evenodd"
                d="M16.22 3H3.78a.78.78 0 00-.78.78v12.44c0 .43.35.78.78.78h12.44c.43 0 .78-.35.78-.78V3.78a.78.78 0 00-.78-.78zM3.78 2h12.44C17.2 2 18 2.8 18 3.78v12.44c0 .98-.8 1.78-1.78 1.78H3.78C2.8 18 2 17.2 2 16.22V3.78C2 2.8 2.8 2 3.78 2zM11 9h3v2h-3v3H9v-3H6V9h3V6h2v3z"
            />
        </svg>
    `,
  expandedIcon:
  /* html */
  `
        <svg
            width="20"
            height="20"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                fill-rule="evenodd"
                clip-rule="evenodd"
                d="M3.78 3h12.44c.43 0 .78.35.78.78v12.44c0 .43-.35.78-.78.78H3.78a.78.78 0 01-.78-.78V3.78c0-.43.35-.78.78-.78zm12.44-1H3.78C2.8 2 2 2.8 2 3.78v12.44C2 17.2 2.8 18 3.78 18h12.44c.98 0 1.78-.8 1.78-1.78V3.78C18 2.8 17.2 2 16.22 2zM14 9H6v2h8V9z"
            />
        </svg>
    `
});

/**
 * A function that returns a {@link @microsoft/fast-foundation#Accordion} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#accordionTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-accordion>`
 */

const fastAccordion = Accordion.compose({
  baseName: "accordion",
  template: accordionTemplate,
  styles: accordionStyles
});

/*! *****************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
function __decorate(decorators, target, key, desc) {
  var c = arguments.length,
      r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc,
      d;
  if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
  return c > 3 && r && Object.defineProperty(target, key, r), r;
}

/**
 * Define shadow algorithms.
 *
 * TODO: The --background-luminance will need to be derived from JavaScript. For now
 * this is hard-coded to a 1, the relative luminance of pure white.
 * https://github.com/microsoft/fast/issues/2778
 *
 * @internal
 */
const ambientShadow = "0 0 calc((var(--elevation) * 0.225px) + 2px) rgba(0, 0, 0, calc(.11 * (2 - var(--background-luminance, 1))))";
/**
 * @internal
 */

const directionalShadow = "0 calc(var(--elevation) * 0.4px) calc((var(--elevation) * 0.9px)) rgba(0, 0, 0, calc(.13 * (2 - var(--background-luminance, 1))))";
/**
 * Applies the box-shadow CSS rule set to the elevation formula.
 * Control this formula with the --elevation CSS Custom Property
 * by setting --elevation to a number.
 */

const elevation = `box-shadow: ${ambientShadow}, ${directionalShadow};`;

/**
 * @internal
 */

const BaseButtonStyles = css`
    ${display("inline-flex")} :host{font-family:${bodyFont};outline:none;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};height:calc(${heightNumber} * 1px);min-width:calc(${heightNumber} * 1px);background-color:${neutralFillRest};color:${neutralForegroundRest};border-radius:calc(${controlCornerRadius} * 1px);fill:currentcolor;cursor:pointer}.control{background:transparent;height:inherit;flex-grow:1;box-sizing:border-box;display:inline-flex;justify-content:center;align-items:baseline;padding:0 calc((10 + (${designUnit} * 2 * ${density})) * 1px);white-space:nowrap;outline:none;text-decoration:none;border:calc(${strokeWidth} * 1px) solid transparent;color:inherit;border-radius:inherit;fill:inherit;cursor:inherit;font-weight:inherit;font-family:inherit;font-size:inherit;line-height:inherit}:host(:hover){background-color:${neutralFillHover}}:host(:active){background-color:${neutralFillActive}}.control:${focusVisible}{border-color:${focusStrokeOuter};box-shadow:0 0 0 calc((${focusStrokeWidth} - ${strokeWidth}) * 1px) ${focusStrokeOuter} inset}.control::-moz-focus-inner{border:0}.start,.content,.end{align-self:center}.start,.end{display:flex}.control.icon-only{padding:0;line-height:0}::slotted(svg){${
/* Glyph size and margin-left is temporary -
replace when adaptive typography is figured out */
""} width:16px;height:16px;pointer-events:none}.start{margin-inline-end:11px}.end{margin-inline-start:11px}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host .control{background-color:${SystemColors.ButtonFace};border-color:${SystemColors.ButtonText};color:${SystemColors.ButtonText};fill:currentColor}:host(:hover) .control{forced-color-adjust:none;background-color:${SystemColors.Highlight};color:${SystemColors.HighlightText}}.control:${focusVisible}{forced-color-adjust:none;background-color:${SystemColors.Highlight};border-color:${SystemColors.ButtonText};box-shadow:0 0 0 calc((${focusStrokeWidth} - ${strokeWidth}) * 1px) ${SystemColors.ButtonText} inset;color:${SystemColors.HighlightText}}.control:hover,:host([appearance="outline"]) .control:hover{border-color:${SystemColors.ButtonText}}:host([href]) .control{border-color:${SystemColors.LinkText};color:${SystemColors.LinkText}}:host([href]) .control:hover,:host([href]) .control:${focusVisible}{forced-color-adjust:none;background:${SystemColors.ButtonFace};border-color:${SystemColors.LinkText};box-shadow:0 0 0 1px ${SystemColors.LinkText} inset;color:${SystemColors.LinkText};fill:currentColor}`));
/**
 * @internal
 */

const AccentButtonStyles = css`
    :host([appearance="accent"]){background:${accentFillRest};color:${foregroundOnAccentRest}}:host([appearance="accent"]:hover){background:${accentFillHover};color:${foregroundOnAccentHover}}:host([appearance="accent"]:active) .control:active{background:${accentFillActive};color:${foregroundOnAccentActive}}:host([appearance="accent"]) .control:${focusVisible}{box-shadow:0 0 0 calc((${focusStrokeWidth} - ${strokeWidth}) * 1px) ${focusStrokeOuter} inset,0 0 0 calc((${focusStrokeWidth} + ${strokeWidth}) * 1px) ${focusStrokeInner} inset}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host([appearance="accent"]) .control{forced-color-adjust:none;background:${SystemColors.Highlight};color:${SystemColors.HighlightText}}:host([appearance="accent"]) .control:hover,:host([appearance="accent"]:active) .control:active{background:${SystemColors.HighlightText};border-color:${SystemColors.Highlight};color:${SystemColors.Highlight}}:host([appearance="accent"]) .control:${focusVisible}{border-color:${SystemColors.Highlight};box-shadow:0 0 0 calc(${focusStrokeWidth} * 1px) ${SystemColors.HighlightText} inset}:host([appearance="accent"][href]) .control{background:${SystemColors.LinkText};color:${SystemColors.HighlightText}}:host([appearance="accent"][href]) .control:hover{background:${SystemColors.ButtonFace};border-color:${SystemColors.LinkText};box-shadow:none;color:${SystemColors.LinkText};fill:currentColor}:host([appearance="accent"][href]) .control:${focusVisible}{border-color:${SystemColors.LinkText};box-shadow:0 0 0 calc(${focusStrokeWidth} * 1px) ${SystemColors.HighlightText} inset}`));
/**
 * @internal
 */

const HypertextStyles = css`
    :host([appearance="hypertext"]){font-size:inherit;line-height:inherit;height:auto;min-width:0;background:transparent}:host([appearance="hypertext"]) .control{display:inline;padding:0;border:none;box-shadow:none;border-radius:0;line-height:1}:host a.control:not(:link){background-color:transparent;cursor:default}:host([appearance="hypertext"]) .control:link,:host([appearance="hypertext"]) .control:visited{background:transparent;color:${accentForegroundRest};border-bottom:calc(${strokeWidth} * 1px) solid ${accentForegroundRest}}:host([appearance="hypertext"]:hover),:host([appearance="hypertext"]) .control:hover{background:transparent;border-bottom-color:${accentForegroundHover}}:host([appearance="hypertext"]:active),:host([appearance="hypertext"]) .control:active{background:transparent;border-bottom-color:${accentForegroundActive}}:host([appearance="hypertext"]) .control:${focusVisible}{border-bottom:calc(${focusStrokeWidth} * 1px) solid ${focusStrokeOuter};margin-bottom:calc(calc(${strokeWidth} - ${focusStrokeWidth}) * 1px)}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host([appearance="hypertext"]:hover){background-color:${SystemColors.ButtonFace};color:${SystemColors.ButtonText}}:host([appearance="hypertext"][href]) .control:hover,:host([appearance="hypertext"][href]) .control:active,:host([appearance="hypertext"][href]) .control:${focusVisible}{color:${SystemColors.LinkText};border-bottom-color:${SystemColors.LinkText};box-shadow:none}`));
/**
 * @internal
 */

const LightweightButtonStyles = css`
    :host([appearance="lightweight"]){background:transparent;color:${accentForegroundRest}}:host([appearance="lightweight"]) .control{padding:0;height:initial;border:none;box-shadow:none;border-radius:0}:host([appearance="lightweight"]:hover){background:transparent;color:${accentForegroundHover}}:host([appearance="lightweight"]:active){background:transparent;color:${accentForegroundActive}}:host([appearance="lightweight"]) .content{position:relative}:host([appearance="lightweight"]) .content::before{content:"";display:block;height:calc(${strokeWidth} * 1px);position:absolute;top:calc(1em + 4px);width:100%}:host([appearance="lightweight"]:hover) .content::before{background:${accentForegroundHover}}:host([appearance="lightweight"]:active) .content::before{background:${accentForegroundActive}}:host([appearance="lightweight"]) .control:${focusVisible} .content::before{background:${neutralForegroundRest};height:calc(${focusStrokeWidth} * 1px)}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host([appearance="lightweight"]) .control:hover,:host([appearance="lightweight"]) .control:${focusVisible}{forced-color-adjust:none;background:${SystemColors.ButtonFace};color:${SystemColors.Highlight}}:host([appearance="lightweight"]) .control:hover .content::before,:host([appearance="lightweight"]) .control:${focusVisible} .content::before{background:${SystemColors.Highlight}}:host([appearance="lightweight"][href]) .control:hover,:host([appearance="lightweight"][href]) .control:${focusVisible}{background:${SystemColors.ButtonFace};box-shadow:none;color:${SystemColors.LinkText}}:host([appearance="lightweight"][href]) .control:hover .content::before,:host([appearance="lightweight"][href]) .control:${focusVisible} .content::before{background:${SystemColors.LinkText}}`));
/**
 * @internal
 */

const OutlineButtonStyles = css`
    :host([appearance="outline"]){background:transparent;border-color:${accentFillRest}}:host([appearance="outline"]:hover){border-color:${accentFillHover}}:host([appearance="outline"]:active){border-color:${accentFillActive}}:host([appearance="outline"]) .control{border-color:inherit}:host([appearance="outline"]) .control:${focusVisible}{box-shadow:0 0 0 calc((${focusStrokeWidth} - ${strokeWidth}) * 1px) ${focusStrokeOuter} inset;border-color:${focusStrokeOuter}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host([appearance="outline"]) .control{border-color:${SystemColors.ButtonText}}:host([appearance="outline"]) .control:${focusVisible}{forced-color-adjust:none;background-color:${SystemColors.Highlight};border-color:${SystemColors.ButtonText};box-shadow:0 0 0 calc((${focusStrokeWidth} - ${strokeWidth}) * 1px) ${SystemColors.ButtonText} inset;color:${SystemColors.HighlightText};fill:currentColor}:host([appearance="outline"][href]) .control{background:${SystemColors.ButtonFace};border-color:${SystemColors.LinkText};color:${SystemColors.LinkText};fill:currentColor}:host([appearance="outline"][href]) .control:hover,:host([appearance="outline"][href]) .control:${focusVisible}{forced-color-adjust:none;border-color:${SystemColors.LinkText};box-shadow:0 0 0 1px ${SystemColors.LinkText} inset}`));
/**
 * @internal
 */

const StealthButtonStyles = css`
    :host([appearance="stealth"]){background:${neutralFillStealthRest}}:host([appearance="stealth"]:hover){background:${neutralFillStealthHover}}:host([appearance="stealth"]:active){background:${neutralFillStealthActive}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host([appearance="stealth"]),:host([appearance="stealth"]) .control{forced-color-adjust:none;background:${SystemColors.ButtonFace};border-color:transparent;color:${SystemColors.ButtonText};fill:currentColor}:host([appearance="stealth"]:hover) .control{background:${SystemColors.Highlight};border-color:${SystemColors.Highlight};color:${SystemColors.HighlightText};fill:currentColor}:host([appearance="stealth"]:${focusVisible}) .control{background:${SystemColors.Highlight};box-shadow:0 0 0 1px ${SystemColors.Highlight};color:${SystemColors.HighlightText};fill:currentColor}:host([appearance="stealth"][href]) .control{color:${SystemColors.LinkText}}:host([appearance="stealth"][href]:hover) .control,:host([appearance="stealth"][href]:${focusVisible}) .control{background:${SystemColors.LinkText};border-color:${SystemColors.LinkText};color:${SystemColors.HighlightText};fill:currentColor}:host([appearance="stealth"][href]:${focusVisible}) .control{forced-color-adjust:none;box-shadow:0 0 0 1px ${SystemColors.LinkText}}`));

/**
 * Behavior to conditionally apply LTR and RTL stylesheets. To determine which to apply,
 * the behavior will use the nearest DesignSystemProvider's 'direction' design system value.
 *
 * @public
 * @example
 * ```ts
 * import { css } from "@microsoft/fast-element";
 * import { DirectionalStyleSheetBehavior } from "@microsoft/fast-foundation";
 *
 * css`
 *  // ...
 * `.withBehaviors(new DirectionalStyleSheetBehavior(
 *   css`:host { content: "ltr"}`),
 *   css`:host { content: "rtl"}`),
 * )
 * ```
 */

class DirectionalStyleSheetBehavior {
  constructor(ltr, rtl) {
    this.cache = new WeakMap();
    this.ltr = ltr;
    this.rtl = rtl;
  }
  /**
   * @internal
   */


  bind(source) {
    this.attach(source);
  }
  /**
   * @internal
   */


  unbind(source) {
    const cache = this.cache.get(source);

    if (cache) {
      direction.unsubscribe(cache);
    }
  }

  attach(source) {
    const subscriber = this.cache.get(source) || new DirectionalStyleSheetBehaviorSubscription(this.ltr, this.rtl, source);
    const value = direction.getValueFor(source);
    direction.subscribe(subscriber);
    subscriber.attach(value);
    this.cache.set(source, subscriber);
  }

}
/**
 * Subscription for {@link DirectionalStyleSheetBehavior}
 */

class DirectionalStyleSheetBehaviorSubscription {
  constructor(ltr, rtl, source) {
    this.ltr = ltr;
    this.rtl = rtl;
    this.source = source;
    this.attached = null;
  }

  handleChange({
    target,
    token
  }) {
    this.attach(token.getValueFor(target));
  }

  attach(direction) {
    if (this.attached !== this[direction]) {
      if (this.attached !== null) {
        this.source.$fastController.removeStyles(this.attached);
      }

      this.attached = this[direction];

      if (this.attached !== null) {
        this.source.$fastController.addStyles(this.attached);
      }
    }
  }

}

/**
 * Behavior that will conditionally apply a stylesheet based on the elements
 * appearance property
 *
 * @param value - The value of the appearance property
 * @param styles - The styles to be applied when condition matches
 *
 * @public
 */

function appearanceBehavior(value, styles) {
  return new PropertyStyleSheetBehavior("appearance", value, styles);
}

/**
 * Styles for Anchor
 * @public
 */

const anchorStyles = (context, definition) => css`
        ${BaseButtonStyles}
    `.withBehaviors(appearanceBehavior("accent", AccentButtonStyles), appearanceBehavior("hypertext", HypertextStyles), appearanceBehavior("lightweight", LightweightButtonStyles), appearanceBehavior("outline", OutlineButtonStyles), appearanceBehavior("stealth", StealthButtonStyles));

/**
 * Base class for Anchor
 * @public
 */

class Anchor extends Anchor$1 {
  appearanceChanged(oldValue, newValue) {
    if (this.$fastController.isConnected) {
      this.classList.remove(oldValue);
      this.classList.add(newValue);
    }
  }

  connectedCallback() {
    super.connectedCallback();

    if (!this.appearance) {
      this.appearance = "neutral";
    }
  }
  /**
   * Applies 'icon-only' class when there is only an SVG in the default slot
   *
   * @internal
   *
   */


  defaultSlottedContentChanged(oldValue, newValue) {
    const slottedElements = this.defaultSlottedContent.filter(x => x.nodeType === Node.ELEMENT_NODE);

    if (slottedElements.length === 1 && slottedElements[0] instanceof SVGElement) {
      this.control.classList.add("icon-only");
    } else {
      this.control.classList.remove("icon-only");
    }
  }

}

__decorate([attr], Anchor.prototype, "appearance", void 0);
/**
 * A function that returns a {@link @microsoft/fast-foundation#Anchor} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#anchorTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-anchor>`
 *
 * {@link https://developer.mozilla.org/en-US/docs/Web/API/ShadowRoot/delegatesFocus | delegatesFocus}
 */


const fastAnchor = Anchor.compose({
  baseName: "anchor",
  baseClass: Anchor$1,
  template: anchorTemplate,
  styles: anchorStyles,
  shadowOptions: {
    delegatesFocus: true
  }
});

/**
 * Styles for AnchoredRegion
 * @public
 */

const anchoredRegionStyles = (context, definition) => css`
    :host{contain:layout;display:block}`;

/**
 * A function that returns a {@link @microsoft/fast-foundation#AnchoredRegion} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#anchoredRegionTemplate}
 *
 *
 * @beta
 * @remarks
 * Generates HTML Element: `<fast-anchored-region>`
 */

const fastAnchoredRegion = AnchoredRegion.compose({
  baseName: "anchored-region",
  template: anchoredRegionTemplate,
  styles: anchoredRegionStyles
});

const rtl$2 = (context, definition) => css`
    ::slotted(${context.tagFor(Badge)}){left:0}`;

const ltr$2 = (context, definition) => css`
    ::slotted(${context.tagFor(Badge)}){right:0}`;
/**
 * Styles for Avatar
 * @public
 */


const avatarStyles = (context, definition) => css`
        ${display("flex")} :host{position:relative;height:var(--avatar-size,var(--avatar-size-default));max-width:var(--avatar-size,var(--avatar-size-default));--avatar-size-default:calc(
                (
                        (${baseHeightMultiplier} + ${density}) * ${designUnit} +
                            ((${designUnit} * 8) - 40)
                    ) * 1px
            );--avatar-text-size:${typeRampBaseFontSize};--avatar-text-ratio:${designUnit}}.link{text-decoration:none;color:${neutralForegroundRest};display:flex;flex-direction:row;justify-content:center;align-items:center;min-width:100%}.square{border-radius:calc(${controlCornerRadius} * 1px);min-width:100%;overflow:hidden}.circle{border-radius:100%;min-width:100%;overflow:hidden}.backplate{position:relative;display:flex}.media,::slotted(img){max-width:100%;position:absolute;display:block}.content{font-size:calc(
                (var(--avatar-text-size) + var(--avatar-size,var(--avatar-size-default))) /
                    var(--avatar-text-ratio)
            );line-height:var(--avatar-size,var(--avatar-size-default));display:block;min-height:var(--avatar-size,var(--avatar-size-default))}::slotted(${context.tagFor(Badge)}){position:absolute;display:block}`.withBehaviors(new DirectionalStyleSheetBehavior(ltr$2(context), rtl$2(context)));

/**
 * The FAST Avatar Class
 * @public
 *
 */

class Avatar extends Avatar$1 {}

__decorate([attr({
  attribute: "src"
})], Avatar.prototype, "imgSrc", void 0);

__decorate([attr], Avatar.prototype, "alt", void 0);
/**
 * The FAST Avatar Template for Images
 *  @public
 *
 */


const imgTemplate = html` ${when(x => x.imgSrc, html`<img src="${x => x.imgSrc}" alt="${x => x.alt}" slot="media" class="media" part="media" />`)}
`;
/**
 * A function that returns a {@link @microsoft/fast-foundation#Avatar} registration for configuring the component with a DesignSystem.
 *  {@link @microsoft/fast-foundation#avatarTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-avatar>`
 */

const fastAvatar = Avatar.compose({
  baseName: "avatar",
  baseClass: Avatar$1,
  template: avatarTemplate,
  styles: avatarStyles,
  media: imgTemplate,
  shadowOptions: {
    delegatesFocus: true
  }
});

/**
 * Styles for Badge
 * @public
 */

const badgeStyles = (context, definition) => css`
        ${display("inline-block")} :host{box-sizing:border-box;font-family:${bodyFont};font-size:${typeRampMinus1FontSize};line-height:${typeRampMinus1LineHeight}}.control{border-radius:calc(${controlCornerRadius} * 1px);padding:calc(((${designUnit} * 0.5) - ${strokeWidth}) * 1px)
                calc((${designUnit} - ${strokeWidth}) * 1px);color:${accentForegroundRest};font-weight:600;border:calc(${strokeWidth} * 1px) solid transparent}.control[style]{font-weight:400}:host([circular]) .control{border-radius:100px;padding:0 calc(${designUnit} * 1px);height:calc((${heightNumber} - (${designUnit} * 3)) * 1px);min-width:calc((${heightNumber} - (${designUnit} * 3)) * 1px);display:flex;align-items:center;justify-content:center;box-sizing:border-box}`;

/**
 * A function that returns a {@link @microsoft/fast-foundation#Badge} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#badgeTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-badge>`
 */

const fastBadge = Badge.compose({
  baseName: "badge",
  template: badgeTemplate,
  styles: badgeStyles
});

/**
 * Styles for Breadcrumb item
 * @public
 */

const breadcrumbItemStyles = (context, definition) => css`
    ${display("inline-flex")} :host{background:transparent;box-sizing:border-box;font-family:${bodyFont};font-size:${typeRampBaseFontSize};fill:currentColor;line-height:${typeRampBaseLineHeight};min-width:calc(${heightNumber} * 1px);outline:none;color:${neutralForegroundRest}}.listitem{display:flex;align-items:center;width:max-content}.separator{margin:0 6px;display:flex}.control{align-items:center;box-sizing:border-box;color:${accentForegroundRest};cursor:pointer;display:flex;fill:inherit;outline:none;text-decoration:none;white-space:nowrap}.control:hover{color:${accentForegroundHover}}.control:active{color:${accentForegroundActive}}.control .content{position:relative}.control .content::before{content:"";display:block;height:calc(${strokeWidth} * 1px);left:0;position:absolute;right:0;top:calc(1em + 4px);width:100%}.control:hover .content::before{background:${accentForegroundHover}}.control:active .content::before{background:${accentForegroundActive}}.control:${focusVisible} .content::before{background:${neutralForegroundRest};height:calc(${focusStrokeWidth} * 1px)}.control:not([href]){color:${neutralForegroundRest};cursor:default}.control:not([href]) .content::before{background:none}.start,.end{display:flex}::slotted(svg){width:16px;height:16px}.start{margin-inline-end:6px}.end{margin-inline-start:6px}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                .control:hover .content::before,.control:${focusVisible} .content::before{background:${SystemColors.LinkText}}.start,.end{fill:${SystemColors.ButtonText}}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#BreadcrumbItem} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#breadcrumbItemTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-breadcrumb-item>`
 */

const fastBreadcrumbItem = BreadcrumbItem.compose({
  baseName: "breadcrumb-item",
  template: breadcrumbItemTemplate,
  styles: breadcrumbItemStyles,
  separator: "/",
  shadowOptions: {
    delegatesFocus: true
  }
});

/**
 * Styles for Breadcrumb
 * @public
 */

const breadcrumbStyles = (context, definition) => css`
    ${display("inline-block")} :host{box-sizing:border-box;font-family:${bodyFont};font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight}}.list{display:flex;flex-wrap:wrap}`;

/**
 * A function that returns a {@link @microsoft/fast-foundation#Breadcrumb} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#breadcrumbTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-breadcrumb>`
 */

const fastBreadcrumb = Breadcrumb.compose({
  baseName: "breadcrumb",
  template: breadcrumbTemplate,
  styles: breadcrumbStyles
});

/**
 * Styles for Button
 * @public
 */

const buttonStyles = (context, definition) => css`
        :host([disabled]),:host([disabled]:hover),:host([disabled]:active){opacity:${disabledOpacity};background-color:${neutralFillRest};cursor:${disabledCursor}}${BaseButtonStyles}
    `.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host([disabled]),:host([disabled]) .control,:host([disabled]:hover),:host([disabled]:active){forced-color-adjust:none;background-color:${SystemColors.ButtonFace};border-color:${SystemColors.GrayText};color:${SystemColors.GrayText};cursor:${disabledCursor};opacity:1}`), appearanceBehavior("accent", css`
                :host([appearance="accent"][disabled]),:host([appearance="accent"][disabled]:hover),:host([appearance="accent"][disabled]:active){background:${accentFillRest}}${AccentButtonStyles}
            `.withBehaviors(forcedColorsStylesheetBehavior(css`
                        :host([appearance="accent"][disabled]) .control,:host([appearance="accent"][disabled]) .control:hover{background:${SystemColors.ButtonFace};border-color:${SystemColors.GrayText};color:${SystemColors.GrayText}}`))), appearanceBehavior("lightweight", css`
                :host([appearance="lightweight"][disabled]:hover),:host([appearance="lightweight"][disabled]:active){background-color:transparent;color:${accentForegroundRest}}:host([appearance="lightweight"][disabled]) .content::before,:host([appearance="lightweight"][disabled]:hover) .content::before,:host([appearance="lightweight"][disabled]:active) .content::before{background:transparent}${LightweightButtonStyles}
            `.withBehaviors(forcedColorsStylesheetBehavior(css`
                        :host([appearance="lightweight"].disabled) .control{forced-color-adjust:none;color:${SystemColors.GrayText}}:host([appearance="lightweight"].disabled)
                            .control:hover
                            .content::before{background:none}`))), appearanceBehavior("outline", css`
                :host([appearance="outline"][disabled]),:host([appearance="outline"][disabled]:hover),:host([appearance="outline"][disabled]:active){background:transparent;border-color:${accentFillRest}}${OutlineButtonStyles}
            `.withBehaviors(forcedColorsStylesheetBehavior(css`
                        :host([appearance="outline"][disabled]) .control{border-color:${SystemColors.GrayText}}`))), appearanceBehavior("stealth", css`
                :host([appearance="stealth"][disabled]),:host([appearance="stealth"][disabled]:hover),:host([appearance="stealth"][disabled]:active){background:${neutralFillStealthRest}}${StealthButtonStyles}
            `.withBehaviors(forcedColorsStylesheetBehavior(css`
                        :host([appearance="stealth"][disabled]){background:${SystemColors.ButtonFace}}:host([appearance="stealth"][disabled]) .control{background:${SystemColors.ButtonFace};border-color:transparent;color:${SystemColors.GrayText}}`))));

/**
 * @internal
 */

class Button extends Button$1 {
  constructor() {
    super(...arguments);
    /**
     * The appearance the button should have.
     *
     * @public
     * @remarks
     * HTML Attribute: appearance
     */

    this.appearance = "neutral";
  }
  /**
   * Applies 'icon-only' class when there is only an SVG in the default slot
   *
   * @public
   * @remarks
   */


  defaultSlottedContentChanged(oldValue, newValue) {
    const slottedElements = this.defaultSlottedContent.filter(x => x.nodeType === Node.ELEMENT_NODE);

    if (slottedElements.length === 1 && slottedElements[0] instanceof SVGElement) {
      this.control.classList.add("icon-only");
    } else {
      this.control.classList.remove("icon-only");
    }
  }

}

__decorate([attr], Button.prototype, "appearance", void 0);
/**
 * A function that returns a {@link @microsoft/fast-foundation#Button} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#buttonTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-button>`
 *
 * {@link https://developer.mozilla.org/en-US/docs/Web/API/ShadowRoot/delegatesFocus | delegatesFocus}
 */


const fastButton = Button.compose({
  baseName: "button",
  baseClass: Button$1,
  template: buttonTemplate,
  styles: buttonStyles,
  shadowOptions: {
    delegatesFocus: true
  }
});

/**
 * Styles for Calendar
 * @public
 */

const CalendarStyles = css`
    ${display("block")} :host{--cell-border:none;--cell-height:calc(${heightNumber} * 1px);--selected-day-outline:1px solid ${accentForegroundActive};--selected-day-color:${accentForegroundActive};--selected-day-background:${neutralFillRest};--cell-padding:calc(${designUnit} * 1px);--disabled-day-opacity:${disabledOpacity};--inactive-day-opacity:${disabledOpacity};font-family:${bodyFont};font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};color:${neutralForegroundRest}}.title{font-size:${typeRampPlus3FontSize};line-height:${typeRampPlus3LineHeight};padding:var(--cell-padding);text-align:center}.week-days,.week{display:grid;grid-template-columns:repeat(7,1fr);border-left:var(--cell-border,none);border-bottom:none;padding:0}.interact .week{grid-gap:calc(${designUnit} * 1px);margin-top:calc(${designUnit} * 1px)}.day,.week-day{border-bottom:var(--cell-border);border-right:var(--cell-border);padding:var(--cell-padding)}.week-day{text-align:center;border-radius:0;border-top:var(--cell-border)}.day{box-sizing:border-box;vertical-align:top;outline-offset:-1px;line-height:var(--cell-line-height);white-space:normal}.interact .day{background:${neutralFillRest};cursor:pointer}.day.inactive{background:var(--inactive-day-background);color:var(--inactive-day-color);opacity:var(--inactive-day-opacity);outline:var(--inactive-day-outline)}.day.disabled{background:var(--disabled-day-background);color:var(--disabled-day-color);cursor:${disabledCursor};opacity:var(--disabled-day-opacity);outline:var(--disabled-day-outline)}.day.selected{color:var(--selected-day-color);background:var(--selected-day-background);outline:var(--selected-day-outline)}.date{padding:var(--cell-padding);text-align:center}.interact .today,.today{color:${foregroundOnAccentActive};background:${accentForegroundActive}}.today.inactive .date{background:transparent;color:inherit;width:auto}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host{--selected-day-outline:1px solid ${SystemColors.Highlight}}.day,.week-day{background:${SystemColors.Canvas};color:${SystemColors.CanvasText};fill:currentcolor}.day.selected{color:${SystemColors.Highlight}}.today .date{background:${SystemColors.Highlight};color:${SystemColors.HighlightText}}`));

/**
 * The FAST Calendar Element. Implements {@link @microsoft/fast-foundation#Calendar},
 * {@link @microsoft/fast-foundation#calendarTemplate}
 *
 *
 * @public
 * @remarks
 * HTML Element: `<fast-calendar>`
 */

const fastCalendar = Calendar.compose({
  baseName: "calendar",
  template: calendarTemplate,
  styles: CalendarStyles,
  title: CalendarTitleTemplate
});

/**
 * Styles for Card
 * @public
 */

const cardStyles = (context, definition) => css`
        ${display("block")} :host{--elevation:4;display:block;contain:content;height:var(--card-height,100%);width:var(--card-width,100%);box-sizing:border-box;background:${fillColor};border-radius:calc(${controlCornerRadius} * 1px);${elevation}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host{forced-color-adjust:none;background:${SystemColors.Canvas};box-shadow:0 0 0 1px ${SystemColors.CanvasText}}`));

/**
 * @internal
 */

class Card extends Card$1 {
  connectedCallback() {
    super.connectedCallback();
    const parent = composedParent(this);

    if (parent) {
      fillColor.setValueFor(this, target => neutralFillLayerRecipe.getValueFor(target).evaluate(target, fillColor.getValueFor(parent)));
    }
  }

}
/**
 * A function that returns a {@link @microsoft/fast-foundation#Card} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#cardTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-card>`
 */

const fastCard = Card.compose({
  baseName: "card",
  baseClass: Card$1,
  template: cardTemplate,
  styles: cardStyles
});

/**
 * Styles for Checkbox
 * @public
 */

const checkboxStyles = (context, definition) => css`
        ${display("inline-flex")} :host{align-items:center;outline:none;margin:calc(${designUnit} * 1px) 0;user-select:none}.control{position:relative;width:calc((${heightNumber} / 2 + ${designUnit}) * 1px);height:calc((${heightNumber} / 2 + ${designUnit}) * 1px);box-sizing:border-box;border-radius:calc(${controlCornerRadius} * 1px);border:calc(${strokeWidth} * 1px) solid ${neutralStrokeRest};background:${neutralFillInputRest};outline:none;cursor:pointer}.label{font-family:${bodyFont};color:${neutralForegroundRest};padding-inline-start:calc(${designUnit} * 2px + 2px);margin-inline-end:calc(${designUnit} * 2px + 2px);cursor:pointer;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight}}.label__hidden{display:none;visibility:hidden}.checked-indicator{width:100%;height:100%;display:block;fill:${foregroundOnAccentRest};opacity:0;pointer-events:none}.indeterminate-indicator{border-radius:calc(${controlCornerRadius} * 1px);background:${foregroundOnAccentRest};position:absolute;top:50%;left:50%;width:50%;height:50%;transform:translate(-50%,-50%);opacity:0}:host(:not([disabled])) .control:hover{background:${neutralFillInputHover};border-color:${neutralStrokeHover}}:host(:not([disabled])) .control:active{background:${neutralFillInputActive};border-color:${neutralStrokeActive}}:host(:${focusVisible}) .control{box-shadow:0 0 0 2px ${fillColor},0 0 0 4px ${focusStrokeOuter}}:host([aria-checked="true"]) .control{background:${accentFillRest};border:calc(${strokeWidth} * 1px) solid ${accentFillRest}}:host([aria-checked="true"]:not([disabled])) .control:hover{background:${accentFillHover};border:calc(${strokeWidth} * 1px) solid ${accentFillHover}}:host([aria-checked="true"]:not([disabled])) .control:hover .checked-indicator{fill:${foregroundOnAccentHover}}:host([aria-checked="true"]:not([disabled])) .control:hover .indeterminate-indicator{background:${foregroundOnAccentHover}}:host([aria-checked="true"]:not([disabled])) .control:active{background:${accentFillActive};border:calc(${strokeWidth} * 1px) solid ${accentFillActive}}:host([aria-checked="true"]:not([disabled])) .control:active .checked-indicator{fill:${foregroundOnAccentActive}}:host([aria-checked="true"]:not([disabled])) .control:active .indeterminate-indicator{background:${foregroundOnAccentActive}}:host([aria-checked="true"]:${focusVisible}:not([disabled])) .control{box-shadow:0 0 0 2px ${fillColor},0 0 0 4px ${focusStrokeOuter}}:host([disabled]) .label,:host([readonly]) .label,:host([readonly]) .control,:host([disabled]) .control{cursor:${disabledCursor}}:host([aria-checked="true"]:not(.indeterminate)) .checked-indicator,:host(.indeterminate) .indeterminate-indicator{opacity:1}:host([disabled]){opacity:${disabledOpacity}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            .control{forced-color-adjust:none;border-color:${SystemColors.FieldText};background:${SystemColors.Field}}.checked-indicator{fill:${SystemColors.FieldText}}.indeterminate-indicator{background:${SystemColors.FieldText}}:host(:not([disabled])) .control:hover,.control:active{border-color:${SystemColors.Highlight};background:${SystemColors.Field}}:host(:${focusVisible}) .control{box-shadow:0 0 0 2px ${SystemColors.Field},0 0 0 4px ${SystemColors.FieldText}}:host([aria-checked="true"]:${focusVisible}:not([disabled])) .control{box-shadow:0 0 0 2px ${SystemColors.Field},0 0 0 4px ${SystemColors.FieldText}}:host([aria-checked="true"]) .control{background:${SystemColors.Highlight};border-color:${SystemColors.Highlight}}:host([aria-checked="true"]:not([disabled])) .control:hover,.control:active{border-color:${SystemColors.Highlight};background:${SystemColors.HighlightText}}:host([aria-checked="true"]) .checked-indicator{fill:${SystemColors.HighlightText}}:host([aria-checked="true"]:not([disabled])) .control:hover .checked-indicator{fill:${SystemColors.Highlight}}:host([aria-checked="true"]) .indeterminate-indicator{background:${SystemColors.HighlightText}}:host([aria-checked="true"]) .control:hover .indeterminate-indicator{background:${SystemColors.Highlight}}:host([disabled]){opacity:1}:host([disabled]) .control{forced-color-adjust:none;border-color:${SystemColors.GrayText};background:${SystemColors.Field}}:host([disabled]) .indeterminate-indicator,:host([aria-checked="true"][disabled]) .control:hover .indeterminate-indicator{forced-color-adjust:none;background:${SystemColors.GrayText}}:host([disabled]) .checked-indicator,:host([aria-checked="true"][disabled]) .control:hover .checked-indicator{forced-color-adjust:none;fill:${SystemColors.GrayText}}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#Checkbox} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#checkboxTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-checkbox>`
 */

const fastCheckbox = Checkbox.compose({
  baseName: "checkbox",
  template: checkboxTemplate,
  styles: checkboxStyles,
  checkedIndicator:
  /* html */
  `
        <svg
            part="checked-indicator"
            class="checked-indicator"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                fill-rule="evenodd"
                clip-rule="evenodd"
                d="M8.143 12.6697L15.235 4.5L16.8 5.90363L8.23812 15.7667L3.80005 11.2556L5.27591 9.7555L8.143 12.6697Z"
            />
        </svg>
    `,
  indeterminateIndicator:
  /* html */
  `
        <div part="indeterminate-indicator" class="indeterminate-indicator"></div>
    `
});

/**
 * Styles for Listbox
 * @public
 */

const listboxStyles = (context, definition) => {
  const ListboxOptionTag = context.tagFor(ListboxOption);
  const hostContext = context.name === context.tagFor(ListboxElement) ? "" : ".listbox"; // The expression interpolations present in this block cause Prettier to generate
  // various formatting bugs.
  // prettier-ignore

  return css`
        ${!hostContext ? display("inline-flex") : ""}

        :host ${hostContext}{background:${fillColor};border:calc(${strokeWidth} * 1px) solid ${neutralStrokeRest};border-radius:calc(${controlCornerRadius} * 1px);box-sizing:border-box;flex-direction:column;padding:calc(${designUnit} * 1px) 0}${!hostContext ? css`
            :host(:focus-within:not([disabled])){border-color:${focusStrokeOuter};box-shadow:0 0 0
                    calc((${focusStrokeWidth} - ${strokeWidth}) * 1px)
                    ${focusStrokeOuter} inset}:host([disabled]) ::slotted(*){cursor:${disabledCursor};opacity:${disabledOpacity};pointer-events:none}` : ``}

        ${hostContext || `:host([size])`}{max-height:calc(
                (var(--size) * ${heightNumber} + (${designUnit} * ${strokeWidth} * 2)) * 1px
            );overflow-y:auto}:host([size="0"]) ${hostContext}{max-height:none}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host(:not([multiple]):${focusVisible}) ::slotted(${ListboxOptionTag}[aria-selected="true"]),:host([multiple]:${focusVisible}) ::slotted(${ListboxOptionTag}[aria-checked="true"]){border-color:${SystemColors.ButtonText};box-shadow:0 0 0 calc(${focusStrokeWidth} * 1px) inset ${SystemColors.HighlightText}}:host(:not([multiple]):${focusVisible}) ::slotted(${ListboxOptionTag}[aria-selected="true"]){background:${SystemColors.Highlight};color:${SystemColors.HighlightText};fill:currentcolor}::slotted(${ListboxOptionTag}[aria-selected="true"]:not([aria-checked="true"])){background:${SystemColors.Highlight};border-color:${SystemColors.HighlightText};color:${SystemColors.HighlightText}}`));
};

/**
 * Styles for Select.
 *
 * @public
 */

const selectStyles = (context, definition) => {
  const selectContext = context.name === context.tagFor(Select$1); // The expression interpolations present in this block cause Prettier to generate
  // various formatting bugs.
  // prettier-ignore

  return css`
        ${display("inline-flex")}

        :host{--elevation:14;background:${neutralFillInputRest};border-radius:calc(${controlCornerRadius} * 1px);border:calc(${strokeWidth} * 1px) solid ${accentFillRest};box-sizing:border-box;color:${neutralForegroundRest};font-family:${bodyFont};height:calc(${heightNumber} * 1px);position:relative;user-select:none;min-width:250px;outline:none;vertical-align:top}${selectContext ? css`
            :host(:not([aria-haspopup])){--elevation:0;border:0;height:auto;min-width:0}` : ""}

        ${listboxStyles(context)}

        :host .listbox{${elevation}
            border:none;display:flex;left:0;position:absolute;width:100%;z-index:1}.control + .listbox{--stroke-size:calc(${designUnit} * ${strokeWidth} * 2);max-height:calc(
                (var(--listbox-max-height) * ${heightNumber} + var(--stroke-size)) * 1px
            )}${selectContext ? css`
            :host(:not([aria-haspopup])) .listbox{left:auto;position:static;z-index:auto}` : ""}

        .listbox[hidden]{display:none}.control{align-items:center;box-sizing:border-box;cursor:pointer;display:flex;font-size:${typeRampBaseFontSize};font-family:inherit;line-height:${typeRampBaseLineHeight};min-height:100%;padding:0 calc(${designUnit} * 2.25px);width:100%}:host(:not([disabled]):hover){background:${neutralFillInputHover};border-color:${accentFillHover}}:host(:${focusVisible}){border-color:${focusStrokeOuter}}:host(:not([size]):not([multiple]):not([open]):${focusVisible}),:host([multiple]:${focusVisible}),:host([size]:${focusVisible}){box-shadow:0 0 0 calc(${focusStrokeWidth} * 1px) ${focusStrokeOuter}}:host(:not([multiple]):not([size]):${focusVisible}) ::slotted(${context.tagFor(ListboxOption)}[aria-selected="true"]:not([disabled])){box-shadow:0 0 0 calc(${focusStrokeWidth} * 1px) inset ${focusStrokeInner};border-color:${focusStrokeOuter};background:${accentFillFocus};color:${foregroundOnAccentFocus}}:host([disabled]){cursor:${disabledCursor};opacity:${disabledOpacity}}:host([disabled]) .control{cursor:${disabledCursor};user-select:none}:host([disabled]:hover){background:${neutralFillStealthRest};color:${neutralForegroundRest};fill:currentcolor}:host(:not([disabled])) .control:active{background:${neutralFillInputActive};border-color:${accentFillActive};border-radius:calc(${controlCornerRadius} * 1px)}:host([open][position="above"]) .listbox{border-bottom-left-radius:0;border-bottom-right-radius:0;border-bottom:0;bottom:calc(${heightNumber} * 1px)}:host([open][position="below"]) .listbox{border-top-left-radius:0;border-top-right-radius:0;border-top:0;top:calc(${heightNumber} * 1px)}.selected-value{flex:1 1 auto;font-family:inherit;min-width:calc(var(--listbox-scroll-width,0) - (${designUnit} * 4) * 1px);overflow:hidden;text-align:start;text-overflow:ellipsis;white-space:nowrap}.indicator{flex:0 0 auto;margin-inline-start:1em}slot[name="listbox"]{display:none;width:100%}:host([open]) slot[name="listbox"]{display:flex;position:absolute;${elevation}}.end{margin-inline-start:auto}.start,.end,.indicator,.select-indicator,::slotted(svg){fill:currentcolor;height:1em;min-height:calc(${designUnit} * 4px);min-width:calc(${designUnit} * 4px);width:1em}::slotted([role="option"]),::slotted(option){flex:0 0 auto}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host(:not([disabled]):hover),:host(:not([disabled]):active){border-color:${SystemColors.Highlight}}:host(:not([disabled]):${focusVisible}){background-color:${SystemColors.ButtonFace};box-shadow:0 0 0 calc(${focusStrokeWidth} * 1px) ${SystemColors.Highlight};color:${SystemColors.ButtonText};fill:currentcolor;forced-color-adjust:none}:host(:not([disabled]):${focusVisible}) .listbox{background:${SystemColors.ButtonFace}}:host([disabled]){border-color:${SystemColors.GrayText};background-color:${SystemColors.ButtonFace};color:${SystemColors.GrayText};fill:currentcolor;opacity:1;forced-color-adjust:none}:host([disabled]:hover){background:${SystemColors.ButtonFace}}:host([disabled]) .control{color:${SystemColors.GrayText};border-color:${SystemColors.GrayText}}:host([disabled]) .control .select-indicator{fill:${SystemColors.GrayText}}:host(:${focusVisible}) ::slotted([aria-selected="true"][role="option"]),:host(:${focusVisible}) ::slotted(option[aria-selected="true"]),:host(:${focusVisible}) ::slotted([aria-selected="true"][role="option"]:not([disabled])){background:${SystemColors.Highlight};border-color:${SystemColors.ButtonText};box-shadow:0 0 0 calc(${focusStrokeWidth} * 1px) inset ${SystemColors.HighlightText};color:${SystemColors.HighlightText};fill:currentcolor}.start,.end,.indicator,.select-indicator,::slotted(svg){color:${SystemColors.ButtonText};fill:currentcolor}`));
};

/**
 * Styles for Combobox
 * @public
 */

const comboboxStyles = (context, definition) => css`
    ${selectStyles(context)}

    :host(:empty) .listbox{display:none}:host([disabled]) *,:host([disabled]){cursor:${disabledCursor};user-select:none}.selected-value{-webkit-appearance:none;background:transparent;border:none;color:inherit;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};height:calc(100% - (${strokeWidth} * 1px));margin:auto 0;width:100%}.selected-value:hover,.selected-value:${focusVisible},.selected-value:disabled,.selected-value:active{outline:none}`;

/**
 * Base class for Combobox.
 * @public
 */

class Combobox extends Combobox$1 {
  /**
   * @internal
   */
  maxHeightChanged(prev, next) {
    this.updateComputedStylesheet();
  }
  /**
   * Updates an internal stylesheet with calculated CSS custom properties.
   *
   * @internal
   */


  updateComputedStylesheet() {
    if (this.computedStylesheet) {
      this.$fastController.removeStyles(this.computedStylesheet);
    }

    const popupMaxHeight = Math.floor(this.maxHeight / heightNumberAsToken.getValueFor(this)).toString();
    this.computedStylesheet = css`
            :host{--listbox-max-height:${popupMaxHeight}}`;
    this.$fastController.addStyles(this.computedStylesheet);
  }

}
/**
 * A function that returns a {@link @microsoft/fast-foundation#Combobox} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#comboboxTemplate}
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-combobox>`
 *
 */

const fastCombobox = Combobox.compose({
  baseName: "combobox",
  baseClass: Combobox$1,
  template: comboboxTemplate,
  styles: comboboxStyles,
  shadowOptions: {
    delegatesFocus: true
  },
  indicator:
  /* html */
  `
        <svg
            class="select-indicator"
            part="select-indicator"
            viewBox="0 0 12 7"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                d="M11.85.65c.2.2.2.5 0 .7L6.4 6.84a.55.55 0 01-.78 0L.14 1.35a.5.5 0 11.71-.7L6 5.8 11.15.65c.2-.2.5-.2.7 0z"
            />
        </svg>
    `
});

/**
 * Styles for Data Grid
 * @public
 */

const dataGridStyles = (context, definition) => css`
    :host{display:flex;position:relative;flex-direction:column}`;

/**
 * Styles for Data Grid row
 * @public
 */

const dataGridRowStyles = (context, definition) => css`
    :host{display:grid;padding:1px 0;box-sizing:border-box;width:100%;border-bottom:calc(${strokeWidth} * 1px) solid ${neutralStrokeDividerRest}}:host(.header){}:host(.sticky-header){background:${neutralFillRest};position:sticky;top:0}`;

/**
 * Styles for Data Grid cell
 * @public
 */

const dataGridCellStyles = (context, definition) => css`
        :host{padding:calc(${designUnit} * 1px) calc(${designUnit} * 3px);color:${neutralForegroundRest};box-sizing:border-box;font-family:${bodyFont};font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};font-weight:400;border:transparent calc(${focusStrokeWidth} * 1px) solid;overflow:hidden;white-space:nowrap;border-radius:calc(${controlCornerRadius} * 1px)}:host(.column-header){font-weight:600}:host(:${focusVisible}){border:${focusStrokeOuter} calc(${focusStrokeWidth} * 1px) solid;outline:none;color:${neutralForegroundRest}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
        :host{forced-color-adjust:none;border-color:transparent;background:${SystemColors.Field};color:${SystemColors.FieldText}}:host(:${focusVisible}){border-color:${SystemColors.FieldText};box-shadow:0 0 0 2px inset ${SystemColors.Field};color:${SystemColors.FieldText}}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#DataGridCell} registration for configuring the component with a DesignSystem.
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-data-grid-cell>`
 */

const fastDataGridCell = DataGridCell.compose({
  baseName: "data-grid-cell",
  template: dataGridCellTemplate,
  styles: dataGridCellStyles
});
/**
 * A function that returns a {@link @microsoft/fast-foundation#DataGridRow} registration for configuring the component with a DesignSystem.
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-data-grid-row>`
 */

const fastDataGridRow = DataGridRow.compose({
  baseName: "data-grid-row",
  template: dataGridRowTemplate,
  styles: dataGridRowStyles
});
/**
 * A function that returns a {@link @microsoft/fast-foundation#DataGrid} registration for configuring the component with a DesignSystem.
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-data-grid>`
 */

const fastDataGrid = DataGrid.compose({
  baseName: "data-grid",
  template: dataGridTemplate,
  styles: dataGridStyles
});

/**
 * A {@link ValueConverter} that converts to and from `Swatch` values.
 * @remarks
 * This converter allows for colors represented as string hex values, returning `null` if the
 * input was `null` or `undefined`.
 * @internal
 */

const swatchConverter = {
  toView(value) {
    if (value === null || value === undefined) {
      return null;
    }

    return value === null || value === void 0 ? void 0 : value.toColorString();
  },

  fromView(value) {
    if (value === null || value === undefined) {
      return null;
    }

    const color = parseColorHexRGB(value);
    return color ? SwatchRGB.create(color.r, color.g, color.b) : null;
  }

};
const backgroundStyles = css`
    :host{background-color:${fillColor};color:${neutralForegroundRest}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host{background-color:${SystemColors.ButtonFace};box-shadow:0 0 0 1px ${SystemColors.CanvasText};color:${SystemColors.ButtonText}}`));

function designToken(token) {
  return (source, key) => {
    source[key + "Changed"] = function (prev, next) {
      if (next !== undefined && next !== null) {
        token.setValueFor(this, next);
      } else {
        token.deleteValueFor(this);
      }
    };
  };
}
/**
 * The FAST DesignSystemProvider Element.
 * @internal
 */


class DesignSystemProvider extends FoundationElement {
  constructor() {
    super();
    /**
     * Used to instruct the FASTDesignSystemProvider
     * that it should not set the CSS
     * background-color and color properties
     *
     * @remarks
     * HTML boolean attribute: no-paint
     */

    this.noPaint = false; // If fillColor or baseLayerLuminance change, we need to
    // re-evaluate whether we should have paint styles applied

    const subscriber = {
      handleChange: this.noPaintChanged.bind(this)
    };
    Observable.getNotifier(this).subscribe(subscriber, "fillColor");
    Observable.getNotifier(this).subscribe(subscriber, "baseLayerLuminance");
  }

  noPaintChanged() {
    if (!this.noPaint && (this.fillColor !== void 0 || this.baseLayerLuminance)) {
      this.$fastController.addStyles(backgroundStyles);
    } else {
      this.$fastController.removeStyles(backgroundStyles);
    }
  }

}

__decorate([attr({
  attribute: "no-paint",
  mode: "boolean"
})], DesignSystemProvider.prototype, "noPaint", void 0);

__decorate([attr({
  attribute: "fill-color",
  converter: swatchConverter
}), designToken(fillColor)], DesignSystemProvider.prototype, "fillColor", void 0);

__decorate([attr({
  attribute: "accent-color",
  converter: swatchConverter,
  mode: "fromView"
}), designToken(accentColor)], DesignSystemProvider.prototype, "accentColor", void 0);

__decorate([attr({
  attribute: "neutral-color",
  converter: swatchConverter,
  mode: "fromView"
}), designToken(neutralColor)], DesignSystemProvider.prototype, "neutralColor", void 0);

__decorate([attr({
  converter: nullableNumberConverter
}), designToken(density)], DesignSystemProvider.prototype, "density", void 0);

__decorate([attr({
  attribute: "design-unit",
  converter: nullableNumberConverter
}), designToken(designUnit)], DesignSystemProvider.prototype, "designUnit", void 0);

__decorate([attr({
  attribute: "direction"
}), designToken(direction)], DesignSystemProvider.prototype, "direction", void 0);

__decorate([attr({
  attribute: "base-height-multiplier",
  converter: nullableNumberConverter
}), designToken(baseHeightMultiplier)], DesignSystemProvider.prototype, "baseHeightMultiplier", void 0);

__decorate([attr({
  attribute: "base-horizontal-spacing-multiplier",
  converter: nullableNumberConverter
}), designToken(baseHorizontalSpacingMultiplier)], DesignSystemProvider.prototype, "baseHorizontalSpacingMultiplier", void 0);

__decorate([attr({
  attribute: "control-corner-radius",
  converter: nullableNumberConverter
}), designToken(controlCornerRadius)], DesignSystemProvider.prototype, "controlCornerRadius", void 0);

__decorate([attr({
  attribute: "stroke-width",
  converter: nullableNumberConverter
}), designToken(strokeWidth)], DesignSystemProvider.prototype, "strokeWidth", void 0);

__decorate([attr({
  attribute: "focus-stroke-width",
  converter: nullableNumberConverter
}), designToken(focusStrokeWidth)], DesignSystemProvider.prototype, "focusStrokeWidth", void 0);

__decorate([attr({
  attribute: "disabled-opacity",
  converter: nullableNumberConverter
}), designToken(disabledOpacity)], DesignSystemProvider.prototype, "disabledOpacity", void 0);

__decorate([attr({
  attribute: "type-ramp-minus-2-font-size"
}), designToken(typeRampMinus2FontSize)], DesignSystemProvider.prototype, "typeRampMinus2FontSize", void 0);

__decorate([attr({
  attribute: "type-ramp-minus-2-line-height"
}), designToken(typeRampMinus2LineHeight)], DesignSystemProvider.prototype, "typeRampMinus2LineHeight", void 0);

__decorate([attr({
  attribute: "type-ramp-minus-1-font-size"
}), designToken(typeRampMinus1FontSize)], DesignSystemProvider.prototype, "typeRampMinus1FontSize", void 0);

__decorate([attr({
  attribute: "type-ramp-minus-1-line-height"
}), designToken(typeRampMinus1LineHeight)], DesignSystemProvider.prototype, "typeRampMinus1LineHeight", void 0);

__decorate([attr({
  attribute: "type-ramp-base-font-size"
}), designToken(typeRampBaseFontSize)], DesignSystemProvider.prototype, "typeRampBaseFontSize", void 0);

__decorate([attr({
  attribute: "type-ramp-base-line-height"
}), designToken(typeRampBaseLineHeight)], DesignSystemProvider.prototype, "typeRampBaseLineHeight", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-1-font-size"
}), designToken(typeRampPlus1FontSize)], DesignSystemProvider.prototype, "typeRampPlus1FontSize", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-1-line-height"
}), designToken(typeRampPlus1LineHeight)], DesignSystemProvider.prototype, "typeRampPlus1LineHeight", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-2-font-size"
}), designToken(typeRampPlus2FontSize)], DesignSystemProvider.prototype, "typeRampPlus2FontSize", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-2-line-height"
}), designToken(typeRampPlus2LineHeight)], DesignSystemProvider.prototype, "typeRampPlus2LineHeight", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-3-font-size"
}), designToken(typeRampPlus3FontSize)], DesignSystemProvider.prototype, "typeRampPlus3FontSize", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-3-line-height"
}), designToken(typeRampPlus3LineHeight)], DesignSystemProvider.prototype, "typeRampPlus3LineHeight", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-4-font-size"
}), designToken(typeRampPlus4FontSize)], DesignSystemProvider.prototype, "typeRampPlus4FontSize", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-4-line-height"
}), designToken(typeRampPlus4LineHeight)], DesignSystemProvider.prototype, "typeRampPlus4LineHeight", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-5-font-size"
}), designToken(typeRampPlus5FontSize)], DesignSystemProvider.prototype, "typeRampPlus5FontSize", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-5-line-height"
}), designToken(typeRampPlus5LineHeight)], DesignSystemProvider.prototype, "typeRampPlus5LineHeight", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-6-font-size"
}), designToken(typeRampPlus6FontSize)], DesignSystemProvider.prototype, "typeRampPlus6FontSize", void 0);

__decorate([attr({
  attribute: "type-ramp-plus-6-line-height"
}), designToken(typeRampPlus6LineHeight)], DesignSystemProvider.prototype, "typeRampPlus6LineHeight", void 0);

__decorate([attr({
  attribute: "accent-fill-rest-delta",
  converter: nullableNumberConverter
}), designToken(accentFillRestDelta)], DesignSystemProvider.prototype, "accentFillRestDelta", void 0);

__decorate([attr({
  attribute: "accent-fill-hover-delta",
  converter: nullableNumberConverter
}), designToken(accentFillHoverDelta)], DesignSystemProvider.prototype, "accentFillHoverDelta", void 0);

__decorate([attr({
  attribute: "accent-fill-active-delta",
  converter: nullableNumberConverter
}), designToken(accentFillActiveDelta)], DesignSystemProvider.prototype, "accentFillActiveDelta", void 0);

__decorate([attr({
  attribute: "accent-fill-focus-delta",
  converter: nullableNumberConverter
}), designToken(accentFillFocusDelta)], DesignSystemProvider.prototype, "accentFillFocusDelta", void 0);

__decorate([attr({
  attribute: "accent-foreground-rest-delta",
  converter: nullableNumberConverter
}), designToken(accentForegroundRestDelta)], DesignSystemProvider.prototype, "accentForegroundRestDelta", void 0);

__decorate([attr({
  attribute: "accent-foreground-hover-delta",
  converter: nullableNumberConverter
}), designToken(accentForegroundHoverDelta)], DesignSystemProvider.prototype, "accentForegroundHoverDelta", void 0);

__decorate([attr({
  attribute: "accent-foreground-active-delta",
  converter: nullableNumberConverter
}), designToken(accentForegroundActiveDelta)], DesignSystemProvider.prototype, "accentForegroundActiveDelta", void 0);

__decorate([attr({
  attribute: "accent-foreground-focus-delta",
  converter: nullableNumberConverter
}), designToken(accentForegroundFocusDelta)], DesignSystemProvider.prototype, "accentForegroundFocusDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-rest-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillRestDelta)], DesignSystemProvider.prototype, "neutralFillRestDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-hover-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillHoverDelta)], DesignSystemProvider.prototype, "neutralFillHoverDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-active-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillActiveDelta)], DesignSystemProvider.prototype, "neutralFillActiveDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-focus-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillFocusDelta)], DesignSystemProvider.prototype, "neutralFillFocusDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-input-rest-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillInputRestDelta)], DesignSystemProvider.prototype, "neutralFillInputRestDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-input-hover-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillInputHoverDelta)], DesignSystemProvider.prototype, "neutralFillInputHoverDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-input-active-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillInputActiveDelta)], DesignSystemProvider.prototype, "neutralFillInputActiveDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-input-focus-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillInputFocusDelta)], DesignSystemProvider.prototype, "neutralFillInputFocusDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-stealth-rest-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillStealthRestDelta)], DesignSystemProvider.prototype, "neutralFillStealthRestDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-stealth-hover-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillStealthHoverDelta)], DesignSystemProvider.prototype, "neutralFillStealthHoverDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-stealth-active-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillStealthActiveDelta)], DesignSystemProvider.prototype, "neutralFillStealthActiveDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-stealth-focus-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillStealthFocusDelta)], DesignSystemProvider.prototype, "neutralFillStealthFocusDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-strong-hover-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillStrongHoverDelta)], DesignSystemProvider.prototype, "neutralFillStrongHoverDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-strong-active-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillStrongActiveDelta)], DesignSystemProvider.prototype, "neutralFillStrongActiveDelta", void 0);

__decorate([attr({
  attribute: "neutral-fill-strong-focus-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillStrongFocusDelta)], DesignSystemProvider.prototype, "neutralFillStrongFocusDelta", void 0);

__decorate([attr({
  attribute: "base-layer-luminance",
  converter: nullableNumberConverter
}), designToken(baseLayerLuminance)], DesignSystemProvider.prototype, "baseLayerLuminance", void 0);

__decorate([attr({
  attribute: "neutral-fill-layer-rest-delta",
  converter: nullableNumberConverter
}), designToken(neutralFillLayerRestDelta)], DesignSystemProvider.prototype, "neutralFillLayerRestDelta", void 0);

__decorate([attr({
  attribute: "neutral-stroke-divider-rest-delta",
  converter: nullableNumberConverter
}), designToken(neutralStrokeDividerRestDelta)], DesignSystemProvider.prototype, "neutralStrokeDividerRestDelta", void 0);

__decorate([attr({
  attribute: "neutral-stroke-rest-delta",
  converter: nullableNumberConverter
}), designToken(neutralStrokeRestDelta)], DesignSystemProvider.prototype, "neutralStrokeRestDelta", void 0);

__decorate([attr({
  attribute: "neutral-stroke-hover-delta",
  converter: nullableNumberConverter
}), designToken(neutralStrokeHoverDelta)], DesignSystemProvider.prototype, "neutralStrokeHoverDelta", void 0);

__decorate([attr({
  attribute: "neutral-stroke-active-delta",
  converter: nullableNumberConverter
}), designToken(neutralStrokeActiveDelta)], DesignSystemProvider.prototype, "neutralStrokeActiveDelta", void 0);

__decorate([attr({
  attribute: "neutral-stroke-focus-delta",
  converter: nullableNumberConverter
}), designToken(neutralStrokeFocusDelta)], DesignSystemProvider.prototype, "neutralStrokeFocusDelta", void 0);
/**
 * Template for DesignSystemProvider.
 * @public
 */


const designSystemProviderTemplate = (context, definition) => html`<slot></slot>`;
/**
 * Styles for DesignSystemProvider.
 * @public
 */

const designSystemProviderStyles = (context, definition) => css`
    ${display("block")}
`;
/**
 * A function that returns a {@link DesignSystemProvider} registration for configuring the component with a DesignSystem.
 * @public
 * @remarks
 * Generates HTML Element: `<fast-design-system-provider>`
 */

const fastDesignSystemProvider = DesignSystemProvider.compose({
  baseName: "design-system-provider",
  template: designSystemProviderTemplate,
  styles: designSystemProviderStyles
});

/**
 * Styles for Dialog
 * @public
 */

const dialogStyles = (context, definition) => css`
    :host([hidden]){display:none}:host{--elevation:14;--dialog-height:480px;--dialog-width:640px;display:block}.overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.3);touch-action:none}.positioning-region{display:flex;justify-content:center;position:fixed;top:0;bottom:0;left:0;right:0;overflow:auto}.control{${elevation}
        margin-top:auto;margin-bottom:auto;width:var(--dialog-width);height:var(--dialog-height);background-color:${fillColor};z-index:1;border-radius:calc(${controlCornerRadius} * 1px);border:calc(${strokeWidth} * 1px) solid transparent}`;

/**
 * A function that returns a {@link @microsoft/fast-foundation#Dialog} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#dialogTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-dialog>`
 */

const fastDialog = Dialog.compose({
  baseName: "dialog",
  template: dialogTemplate,
  styles: dialogStyles
});

/**
 * Styles for Disclosure
 * @public
 */

const disclosureStyles = (context, definition) => css`
    .disclosure{transition:height 0.35s}.disclosure .invoker::-webkit-details-marker{display:none}.disclosure .invoker{list-style-type:none}:host([appearance="accent"]) .invoker{background:${accentFillRest};color:${foregroundOnAccentRest};font-family:${bodyFont};font-size:${typeRampBaseFontSize};border-radius:calc(${controlCornerRadius} * 1px);outline:none;cursor:pointer;margin:16px 0;padding:12px;max-width:max-content}:host([appearance="accent"]) .invoker:active{background:${accentFillActive};color:${foregroundOnAccentActive}}:host([appearance="accent"]) .invoker:hover{background:${accentFillHover};color:${foregroundOnAccentHover}}:host([appearance="lightweight"]) .invoker{background:transparent;color:${accentForegroundRest};border-bottom:calc(${strokeWidth} * 1px) solid ${accentForegroundRest};cursor:pointer;width:max-content;margin:16px 0}:host([appearance="lightweight"]) .invoker:active{border-bottom-color:${accentForegroundActive}}:host([appearance="lightweight"]) .invoker:hover{border-bottom-color:${accentForegroundHover}}.disclosure[open] .invoker ~ *{animation:fadeIn 0.5s ease-in-out}@keyframes fadeIn{0%{opacity:0}100%{opacity:1}}`;

/**
 * @internal
 */

class Disclosure extends Disclosure$1 {
  constructor() {
    super(...arguments);
    /**
     * Disclosure default height
     */

    this.height = 0;
    /**
     * Disclosure height after it's expanded
     */

    this.totalHeight = 0;
  }

  connectedCallback() {
    super.connectedCallback();

    if (!this.appearance) {
      this.appearance = "accent";
    }
  }

  appearanceChanged(oldValue, newValue) {
    if (oldValue !== newValue) {
      this.classList.add(newValue);
      this.classList.remove(oldValue);
    }
  }
  /**
   * Set disclosure height while transitioning
   * @override
   */


  onToggle() {
    super.onToggle();
    this.details.style.setProperty("height", `${this.disclosureHeight}px`);
  }
  /**
   * Calculate disclosure height before and after expanded
   * @override
   */


  setup() {
    super.setup();

    const getCurrentHeight = () => this.details.getBoundingClientRect().height;

    this.show();
    this.totalHeight = getCurrentHeight();
    this.hide();
    this.height = getCurrentHeight();

    if (this.expanded) {
      this.show();
    }
  }

  get disclosureHeight() {
    return this.expanded ? this.totalHeight : this.height;
  }

}

__decorate([attr], Disclosure.prototype, "appearance", void 0);
/**
 * A function that returns a {@link @microsoft/fast-foundation#Disclosure} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#disclosureTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-Disclosure>`
 *
 */


const fastDisclosure = Disclosure.compose({
  baseName: "disclosure",
  baseClass: Disclosure$1,
  template: disclosureTemplate,
  styles: disclosureStyles
});

/**
 * Styles for Divider
 * @public
 */

const dividerStyles = (context, definition) => css`
        ${display("block")} :host{box-sizing:content-box;height:0;margin:calc(${designUnit} * 1px) 0;border-top:calc(${strokeWidth} * 1px) solid ${neutralStrokeDividerRest};border-left:none}:host([orientation="vertical"]){height:100%;margin:0 calc(${designUnit} * 1px);border-top:none;border-left:calc(${strokeWidth} * 1px) solid ${neutralStrokeDividerRest}}`;

/**
 * A function that returns a {@link @microsoft/fast-foundation#Divider} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#dividerTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-divider>`
 */

const fastDivider = Divider.compose({
  baseName: "divider",
  template: dividerTemplate,
  styles: dividerStyles
});

/**
 * Styles for Flipper
 * @public
 */

const flipperStyles = (context, definition) => css`
    ${display("inline-flex")} :host{width:calc(${heightNumber} * 1px);height:calc(${heightNumber} * 1px);justify-content:center;align-items:center;margin:0;position:relative;fill:currentcolor;color:${foregroundOnAccentRest};background:transparent;outline:none;border:none;padding:0}:host::before{content:"";background:${accentFillRest};border:calc(${strokeWidth} * 1px) solid ${accentFillRest};border-radius:50%;position:absolute;top:0;right:0;left:0;bottom:0;transition:all 0.1s ease-in-out}.next,.previous{position:relative;width:16px;height:16px;display:grid}:host([disabled]){opacity:${disabledOpacity};cursor:${disabledCursor};fill:currentcolor;color:${neutralForegroundRest};pointer-events:none}:host([disabled])::before,:host([disabled]:hover)::before,:host([disabled]:active)::before{background:${neutralFillStealthRest};border-color:${neutralStrokeRest}}:host(:hover){color:${foregroundOnAccentHover}}:host(:hover)::before{background:${accentFillHover};border-color:${accentFillHover}}:host(:active){color:${foregroundOnAccentActive}}:host(:active)::before{background:${accentFillActive};border-color:${accentFillActive}}:host(:${focusVisible}){outline:none}:host(:${focusVisible})::before{box-shadow:0 0 0 calc((${focusStrokeWidth} - ${strokeWidth}) * 1px) ${focusStrokeOuter} inset,0 0 0 calc((${focusStrokeWidth} + ${strokeWidth}) * 1px) ${focusStrokeInner} inset;border-color:${focusStrokeOuter}}:host::-moz-focus-inner{border:0}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host{background:${SystemColors.Canvas}}:host .next,:host .previous{color:${SystemColors.ButtonText};fill:currentcolor}:host::before{background:${SystemColors.Canvas};border-color:${SystemColors.ButtonText}}:host(:hover)::before{forced-color-adjust:none;background:${SystemColors.Highlight};border-color:${SystemColors.ButtonText};opacity:1}:host(:hover) .next,:host(:hover) .previous{forced-color-adjust:none;color:${SystemColors.HighlightText};fill:currentcolor}:host([disabled]){opacity:1}:host([disabled])::before,:host([disabled]:hover)::before,:host([disabled]) .next,:host([disabled]) .previous{forced-color-adjust:none;background:${SystemColors.Canvas};border-color:${SystemColors.GrayText};color:${SystemColors.GrayText};fill:${SystemColors.GrayText}}:host(:${focusVisible})::before{forced-color-adjust:none;border-color:${SystemColors.Highlight};box-shadow:0 0 0 calc((${focusStrokeWidth} - ${strokeWidth}) * 1px) ${SystemColors.Highlight} inset}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#Flipper} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#flipperTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-flipper>`
 */

const fastFlipper = Flipper.compose({
  baseName: "flipper",
  template: flipperTemplate,
  styles: flipperStyles,
  next:
  /* html */
  `
        <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
            <path
                d="M4.023 15.273L11.29 8 4.023.727l.704-.704L12.71 8l-7.984 7.977-.704-.704z"
            />
        </svg>
    `,
  previous:
  /* html */
  `
        <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
            <path
                d="M11.273 15.977L3.29 8 11.273.023l.704.704L4.71 8l7.266 7.273-.704.704z"
            />
        </svg>
    `
});

const ltrActionsStyles = css`
    .scroll-prev{right:auto;left:0}.scroll.scroll-next::before,.scroll-next .scroll-action{left:auto;right:0}.scroll.scroll-next::before{background:linear-gradient(to right,transparent,var(--scroll-fade-next))}.scroll-next .scroll-action{transform:translate(50%,-50%)}`;
const rtlActionsStyles = css`
    .scroll.scroll-next{right:auto;left:0}.scroll.scroll-next::before{background:linear-gradient(to right,var(--scroll-fade-next),transparent);left:auto;right:0}.scroll.scroll-prev::before{background:linear-gradient(to right,transparent,var(--scroll-fade-previous))}.scroll-prev .scroll-action{left:auto;right:0;transform:translate(50%,-50%)}`;
/**
 * Styles used for the flipper container and gradient fade actions
 * @public
 */

const ActionsStyles = css`
    .scroll-area{position:relative}div.scroll-view{overflow-x:hidden}.scroll{bottom:0;pointer-events:none;position:absolute;right:0;top:0;user-select:none;width:100px}.scroll.disabled{display:none}.scroll::before,.scroll-action{left:0;position:absolute}.scroll::before{background:linear-gradient(to right,var(--scroll-fade-previous),transparent);content:"";display:block;height:100%;width:100%}.scroll-action{pointer-events:auto;right:auto;top:50%;transform:translate(-50%,-50%)}`.withBehaviors(new DirectionalStyleSheetBehavior(ltrActionsStyles, rtlActionsStyles));
/**
 * Horizontal Scroll styles
 * @public
 */

const horizontalScrollStyles = (context, definition) => css`
    ${display("block")} :host{--scroll-align:center;--scroll-item-spacing:5px;contain:layout;position:relative}.scroll-view{overflow-x:auto;scrollbar-width:none}::-webkit-scrollbar{display:none}.content-container{align-items:var(--scroll-align);display:inline-flex;flex-wrap:nowrap;position:relative}.content-container ::slotted(*){margin-right:var(--scroll-item-spacing)}.content-container ::slotted(*:last-child){margin-right:0}`;

/**
 * @internal
 */

class HorizontalScroll extends HorizontalScroll$1 {
  /**
   * @public
   */
  connectedCallback() {
    super.connectedCallback();

    if (this.view !== "mobile") {
      this.$fastController.addStyles(ActionsStyles);
    }
  }

}
/**
 * A function that returns a {@link @microsoft/fast-foundation#HorizontalScroll} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#horizontalScrollTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-horizontal-scroll>`
 */

const fastHorizontalScroll = HorizontalScroll.compose({
  baseName: "horizontal-scroll",
  baseClass: HorizontalScroll$1,
  template: horizontalScrollTemplate,
  styles: horizontalScrollStyles,
  nextFlipper: context => html`<${context.tagFor(Flipper)} @click="${x => x.scrollToNext()}" aria-hidden="${x => x.flippersHiddenFromAT}"></${context.tagFor(Flipper)}>`,
  previousFlipper: context => html`<${context.tagFor(Flipper)} @click="${x => x.scrollToPrevious()}" direction="previous" aria-hidden="${x => x.flippersHiddenFromAT}"></${context.tagFor(Flipper)}>`
});

/**
 * Styles for the {@link @microsoft/fast-components#fastOption | Listbox Option} component.
 *
 * @param context - the element definition context
 * @param definition - the foundation element definition
 * @returns The element styles for the listbox option component
 *
 * @public
 */

const optionStyles = (context, definition) => css`
        ${display("inline-flex")} :host{align-items:center;font-family:${bodyFont};border-radius:calc(${controlCornerRadius} * 1px);border:calc(${focusStrokeWidth} * 1px) solid transparent;box-sizing:border-box;background:${neutralFillStealthRest};color:${neutralForegroundRest};cursor:pointer;flex:0 0 auto;fill:currentcolor;font-size:${typeRampBaseFontSize};height:calc(${heightNumber} * 1px);line-height:${typeRampBaseLineHeight};margin:0 calc((${designUnit} - ${focusStrokeWidth}) * 1px);outline:none;overflow:hidden;padding:0 1ch;user-select:none;white-space:nowrap}:host(:not([disabled]):not([aria-selected="true"]):hover){background:${neutralFillStealthHover}}:host(:not([disabled]):not([aria-selected="true"]):active){background:${neutralFillStealthActive}}:host([aria-selected="true"]){background:${accentFillRest};color:${foregroundOnAccentRest}}:host(:not([disabled])[aria-selected="true"]:hover){background:${accentFillHover};color:${foregroundOnAccentHover}}:host(:not([disabled])[aria-selected="true"]:active){background:${accentFillActive};color:${foregroundOnAccentActive}}:host([disabled]){cursor:${disabledCursor};opacity:${disabledOpacity}}.content{grid-column-start:2;justify-self:start;overflow:hidden;text-overflow:ellipsis}.start,.end,::slotted(svg){display:flex}::slotted(svg){height:calc(${designUnit} * 4px);width:calc(${designUnit} * 4px)}::slotted([slot="end"]){margin-inline-start:1ch}::slotted([slot="start"]){margin-inline-end:1ch}:host([aria-checked="true"][aria-selected="false"]){border-color:${focusStrokeOuter}}:host([aria-checked="true"][aria-selected="true"]){border-color:${focusStrokeOuter};box-shadow:0 0 0 calc(${focusStrokeWidth} * 2 * 1px) inset
                ${focusStrokeInner}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host{border-color:transparent;forced-color-adjust:none;color:${SystemColors.ButtonText};fill:currentcolor}:host(:not([aria-selected="true"]):hover),:host([aria-selected="true"]){background:${SystemColors.Highlight};color:${SystemColors.HighlightText}}:host([disabled]),:host([disabled][aria-selected="false"]:hover){background:${SystemColors.Canvas};color:${SystemColors.GrayText};fill:currentcolor;opacity:1}:host([aria-checked="true"][aria-selected="false"]){background:${SystemColors.ButtonFace};color:${SystemColors.ButtonText};border-color:${SystemColors.ButtonText}}:host([aria-checked="true"][aria-selected="true"]),:host([aria-checked="true"][aria-selected="true"]:hover){background:${SystemColors.Highlight};color:${SystemColors.HighlightText};border-color:${SystemColors.ButtonText}}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#ListboxOption} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#listboxOptionTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-option>`
 *
 */

const fastOption = ListboxOption.compose({
  baseName: "option",
  template: listboxOptionTemplate,
  styles: optionStyles
});

/**
 * Base class for Listbox.
 *
 * @public
 */

class Listbox extends ListboxElement {
  /**
   * Updates the component dimensions when the size property is changed.
   *
   * @param prev - the previous size value
   * @param next - the current size value
   *
   * @internal
   */
  sizeChanged(prev, next) {
    super.sizeChanged(prev, next);
    this.updateComputedStylesheet();
  }
  /**
   * Updates an internal stylesheet with calculated CSS custom properties.
   *
   * @internal
   */


  updateComputedStylesheet() {
    if (this.computedStylesheet) {
      this.$fastController.removeStyles(this.computedStylesheet);
    }

    const listboxSize = `${this.size}`;
    this.computedStylesheet = css`
            :host{--size:${listboxSize}}`;
    this.$fastController.addStyles(this.computedStylesheet);
  }

}
/**
 * A function that returns a {@link @microsoft/fast-foundation#ListboxElement} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#listboxTemplate}
 *
 * @remarks
 * Generates HTML Element: `<fast-listbox>`
 *
 * @public
 *
 */

const fastListbox = Listbox.compose({
  baseName: "listbox",
  baseClass: ListboxElement,
  template: listboxTemplate,
  styles: listboxStyles
});

/**
 * Styles for Menu item
 * @public
 */

const menuItemStyles = (context, definition) => css`
        ${display("grid")} :host{contain:layout;overflow:visible;font-family:${bodyFont};outline:none;box-sizing:border-box;height:calc(${heightNumber} * 1px);grid-template-columns:minmax(42px,auto) 1fr minmax(42px,auto);grid-template-rows:auto;justify-items:center;align-items:center;padding:0;margin:0 calc(${designUnit} * 1px);white-space:nowrap;background:${neutralFillStealthRest};color:${neutralForegroundRest};fill:currentcolor;cursor:pointer;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};border-radius:calc(${controlCornerRadius} * 1px);border:calc(${focusStrokeWidth} * 1px) solid transparent}:host(:hover){position:relative;z-index:1}:host(.indent-0){grid-template-columns:auto 1fr minmax(42px,auto)}:host(.indent-0) .content{grid-column:1;grid-row:1;margin-inline-start:10px}:host(.indent-0) .expand-collapse-glyph-container{grid-column:5;grid-row:1}:host(.indent-2){grid-template-columns:minmax(42px,auto) minmax(42px,auto) 1fr minmax(42px,auto) minmax(42px,auto)}:host(.indent-2) .content{grid-column:3;grid-row:1;margin-inline-start:10px}:host(.indent-2) .expand-collapse-glyph-container{grid-column:5;grid-row:1}:host(.indent-2) .start{grid-column:2}:host(.indent-2) .end{grid-column:4}:host(:${focusVisible}){border-color:${focusStrokeOuter};background:${neutralFillStealthFocus};color:${neutralForegroundRest}}:host(:hover){background:${neutralFillStealthHover};color:${neutralForegroundRest}}:host(:active){background:${neutralFillStealthActive}}:host([aria-checked="true"]),:host(.expanded){background:${neutralFillRest};color:${neutralForegroundRest}}:host([disabled]){cursor:${disabledCursor};opacity:${disabledOpacity}}:host([disabled]:hover){color:${neutralForegroundRest};fill:currentcolor;background:${neutralFillStealthRest}}:host([disabled]:hover) .start,:host([disabled]:hover) .end,:host([disabled]:hover)::slotted(svg){fill:${neutralForegroundRest}}.expand-collapse-glyph{width:16px;height:16px;fill:currentcolor}.content{grid-column-start:2;justify-self:start;overflow:hidden;text-overflow:ellipsis}.start,.end{display:flex;justify-content:center}::slotted(svg){width:16px;height:16px}:host(:hover) .start,:host(:hover) .end,:host(:hover)::slotted(svg),:host(:active) .start,:host(:active) .end,:host(:active)::slotted(svg){fill:${neutralForegroundRest}}:host(.indent-0[aria-haspopup="menu"]){display:grid;grid-template-columns:minmax(42px,auto) auto 1fr minmax(42px,auto) minmax(42px,auto);align-items:center;min-height:32px}:host(.indent-1[aria-haspopup="menu"]),:host(.indent-1[role="menuitemcheckbox"]),:host(.indent-1[role="menuitemradio"]){display:grid;grid-template-columns:minmax(42px,auto) auto 1fr minmax(42px,auto) minmax(42px,auto);align-items:center;min-height:32px}:host(.indent-2:not([aria-haspopup="menu"])) .end{grid-column:5}:host .input-container,:host .expand-collapse-glyph-container{display:none}:host([aria-haspopup="menu"]) .expand-collapse-glyph-container,:host([role="menuitemcheckbox"]) .input-container,:host([role="menuitemradio"]) .input-container{display:grid;margin-inline-end:10px}:host([aria-haspopup="menu"]) .content,:host([role="menuitemcheckbox"]) .content,:host([role="menuitemradio"]) .content{grid-column-start:3}:host([aria-haspopup="menu"].indent-0) .content{grid-column-start:1}:host([aria-haspopup="menu"]) .end,:host([role="menuitemcheckbox"]) .end,:host([role="menuitemradio"]) .end{grid-column-start:4}:host .expand-collapse,:host .checkbox,:host .radio{display:flex;align-items:center;justify-content:center;position:relative;width:20px;height:20px;box-sizing:border-box;outline:none;margin-inline-start:10px}:host .checkbox,:host .radio{border:calc(${strokeWidth} * 1px) solid ${neutralForegroundRest}}:host([aria-checked="true"]) .checkbox,:host([aria-checked="true"]) .radio{background:${accentFillRest};border-color:${accentFillRest}}:host .checkbox{border-radius:calc(${controlCornerRadius} * 1px)}:host .radio{border-radius:999px}:host .checkbox-indicator,:host .radio-indicator,:host .expand-collapse-indicator,::slotted([slot="checkbox-indicator"]),::slotted([slot="radio-indicator"]),::slotted([slot="expand-collapse-indicator"]){display:none}::slotted([slot="end"]:not(svg)){margin-inline-end:10px;color:${neutralForegroundHint}}:host([aria-checked="true"]) .checkbox-indicator,:host([aria-checked="true"]) ::slotted([slot="checkbox-indicator"]){width:100%;height:100%;display:block;fill:${foregroundOnAccentRest};pointer-events:none}:host([aria-checked="true"]) .radio-indicator{position:absolute;top:4px;left:4px;right:4px;bottom:4px;border-radius:999px;display:block;background:${foregroundOnAccentRest};pointer-events:none}:host([aria-checked="true"]) ::slotted([slot="radio-indicator"]){display:block;pointer-events:none}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host{border-color:transparent;color:${SystemColors.ButtonText};forced-color-adjust:none}:host(:hover){background:${SystemColors.Highlight};color:${SystemColors.HighlightText}}:host(:hover) .start,:host(:hover) .end,:host(:hover)::slotted(svg),:host(:active) .start,:host(:active) .end,:host(:active)::slotted(svg){fill:${SystemColors.HighlightText}}:host(.expanded){background:${SystemColors.Highlight};border-color:${SystemColors.Highlight};color:${SystemColors.HighlightText}}:host(:${focusVisible}){background:${SystemColors.Highlight};border-color:${SystemColors.ButtonText};box-shadow:0 0 0 calc(${focusStrokeWidth} * 1px) inset ${SystemColors.HighlightText};color:${SystemColors.HighlightText};fill:currentcolor}:host([disabled]),:host([disabled]:hover),:host([disabled]:hover) .start,:host([disabled]:hover) .end,:host([disabled]:hover)::slotted(svg){background:${SystemColors.Canvas};color:${SystemColors.GrayText};fill:currentcolor;opacity:1}:host .expanded-toggle,:host .checkbox,:host .radio{border-color:${SystemColors.ButtonText};background:${SystemColors.HighlightText}}:host([checked="true"]) .checkbox,:host([checked="true"]) .radio{background:${SystemColors.HighlightText};border-color:${SystemColors.HighlightText}}:host(:hover) .expanded-toggle,:host(:hover) .checkbox,:host(:hover) .radio,:host(:${focusVisible}) .expanded-toggle,:host(:${focusVisible}) .checkbox,:host(:${focusVisible}) .radio,:host([checked="true"]:hover) .checkbox,:host([checked="true"]:hover) .radio,:host([checked="true"]:${focusVisible}) .checkbox,:host([checked="true"]:${focusVisible}) .radio{border-color:${SystemColors.HighlightText}}:host([aria-checked="true"]){background:${SystemColors.Highlight};color:${SystemColors.HighlightText}}:host([aria-checked="true"]) .checkbox-indicator,:host([aria-checked="true"]) ::slotted([slot="checkbox-indicator"]),:host([aria-checked="true"]) ::slotted([slot="radio-indicator"]){fill:${SystemColors.Highlight}}:host([aria-checked="true"]) .radio-indicator{background:${SystemColors.Highlight}}::slotted([slot="end"]:not(svg)){color:${SystemColors.ButtonText}}:host(:hover) ::slotted([slot="end"]:not(svg)),:host(:${focusVisible}) ::slotted([slot="end"]:not(svg)){color:${SystemColors.HighlightText}}`), new DirectionalStyleSheetBehavior(css`
                .expand-collapse-glyph{transform:rotate(0deg)}`, css`
                .expand-collapse-glyph{transform:rotate(180deg)}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#MenuItem} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#menuItemTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-menu-item>`
 */

const fastMenuItem = MenuItem.compose({
  baseName: "menu-item",
  template: menuItemTemplate,
  styles: menuItemStyles,
  checkboxIndicator:
  /* html */
  `
        <svg
            part="checkbox-indicator"
            class="checkbox-indicator"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                fill-rule="evenodd"
                clip-rule="evenodd"
                d="M8.143 12.6697L15.235 4.5L16.8 5.90363L8.23812 15.7667L3.80005 11.2556L5.27591 9.7555L8.143 12.6697Z"
            />
        </svg>
    `,
  expandCollapseGlyph:
  /* html */
  `
        <svg
            viewBox="0 0 16 16"
            xmlns="http://www.w3.org/2000/svg"
            class="expand-collapse-glyph"
            part="expand-collapse-glyph"
        >
            <path
                d="M5.00001 12.3263C5.00124 12.5147 5.05566 12.699 5.15699 12.8578C5.25831 13.0167 5.40243 13.1437 5.57273 13.2242C5.74304 13.3047 5.9326 13.3354 6.11959 13.3128C6.30659 13.2902 6.4834 13.2152 6.62967 13.0965L10.8988 8.83532C11.0739 8.69473 11.2153 8.51658 11.3124 8.31402C11.4096 8.11146 11.46 7.88966 11.46 7.66499C11.46 7.44033 11.4096 7.21853 11.3124 7.01597C11.2153 6.81341 11.0739 6.63526 10.8988 6.49467L6.62967 2.22347C6.48274 2.10422 6.30501 2.02912 6.11712 2.00691C5.92923 1.9847 5.73889 2.01628 5.56823 2.09799C5.39757 2.17969 5.25358 2.30817 5.153 2.46849C5.05241 2.62882 4.99936 2.8144 5.00001 3.00369V12.3263Z"
            />
        </svg>
    `,
  radioIndicator:
  /* html */
  `
        <span part="radio-indicator" class="radio-indicator"></span>
    `
});

/**
 * Styles for Menu
 * @public
 */

const menuStyles = (context, definition) => css`
        ${display("block")} :host{--elevation:11;background:${fillColor};border:calc(${strokeWidth} * 1px) solid transparent;${elevation}
            margin:0;border-radius:calc(${controlCornerRadius} * 1px);padding:calc(${designUnit} * 1px) 0;max-width:368px;min-width:64px}:host([slot="submenu"]){width:max-content;margin:0 calc(${designUnit} * 1px)}::slotted(hr){box-sizing:content-box;height:0;margin:0;border:none;border-top:calc(${strokeWidth} * 1px) solid ${neutralStrokeDividerRest}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host{background:${SystemColors.Canvas};border-color:${SystemColors.CanvasText}}`));

/**
 * @public
 */

class Menu extends Menu$1 {
  /**
   * @internal
   */
  connectedCallback() {
    super.connectedCallback();
    fillColor.setValueFor(this, neutralLayerFloating);
  }

}
/**
 * A function that returns a {@link @microsoft/fast-foundation#Menu} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#menuTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-menu>`
 */

const fastMenu = Menu.compose({
  baseName: "menu",
  template: menuTemplate,
  styles: menuStyles
});

/**
 * Styles for Number Field
 * @public
 */

const numberFieldStyles = (context, definition) => css`
    ${display("inline-block")} :host{font-family:${bodyFont};outline:none;user-select:none}.root{box-sizing:border-box;position:relative;display:flex;flex-direction:row;color:${neutralForegroundRest};background:${neutralFillInputRest};border-radius:calc(${controlCornerRadius} * 1px);border:calc(${strokeWidth} * 1px) solid ${accentFillRest};height:calc(${heightNumber} * 1px);align-items:baseline}.control{-webkit-appearance:none;font:inherit;background:transparent;border:0;color:inherit;height:calc(100% - 4px);width:100%;margin-top:auto;margin-bottom:auto;border:none;padding:0 calc(${designUnit} * 2px + 1px);font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight}}.control:hover,.control:${focusVisible},.control:disabled,.control:active{outline:none}.controls{opacity:0}.label{display:block;color:${neutralForegroundRest};cursor:pointer;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};margin-bottom:4px}.label__hidden{display:none;visibility:hidden}.start,.control,.controls,.end{align-self:center}.start,.end{margin:auto;fill:currentcolor}.step-up-glyph,.step-down-glyph{display:block;padding:4px 10px;cursor:pointer}.step-up-glyph:before,.step-down-glyph:before{content:'';display:block;border:solid transparent 6px}.step-up-glyph:before{border-bottom-color:${neutralForegroundRest}}.step-down-glyph:before{border-top-color:${neutralForegroundRest}}::slotted(svg){width:16px;height:16px}.start{margin-inline-start:11px}.end{margin-inline-end:11px}:host(:hover:not([disabled])) .root{background:${neutralFillInputHover};border-color:${accentFillHover}}:host(:active:not([disabled])) .root{background:${neutralFillInputHover};border-color:${accentFillActive}}:host(:focus-within:not([disabled])) .root{border-color:${focusStrokeOuter};box-shadow:0 0 0 calc(${focusStrokeWidth} * 1px) ${focusStrokeOuter} inset}:host(:hover:not([disabled])) .controls,:host(:focus-within:not([disabled])) .controls{opacity:1}:host([appearance="filled"]) .root{background:${neutralFillRest}}:host([appearance="filled"]:hover:not([disabled])) .root{background:${neutralFillHover}}:host([disabled]) .label,:host([readonly]) .label,:host([readonly]) .control,:host([disabled]) .control{cursor:${disabledCursor}}:host([disabled]){opacity:${disabledOpacity}}:host([disabled]) .control{border-color:${neutralStrokeRest}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                .root,:host([appearance="filled"]) .root{forced-color-adjust:none;background:${SystemColors.Field};border-color:${SystemColors.FieldText}}:host(:hover:not([disabled])) .root,:host([appearance="filled"]:hover:not([disabled])) .root,:host([appearance="filled"]:hover) .root{background:${SystemColors.Field};border-color:${SystemColors.Highlight}}.start,.end{fill:currentcolor}:host([disabled]){opacity:1}:host([disabled]) .root,:host([appearance="filled"]:hover[disabled]) .root{border-color:${SystemColors.GrayText};background:${SystemColors.Field}}:host(:focus-within:enabled) .root{border-color:${SystemColors.Highlight};box-shadow:0 0 0 1px ${SystemColors.Highlight} inset}input::placeholder{color:${SystemColors.GrayText}}`));

/**
 * @internal
 */

class NumberField extends NumberField$1 {
  constructor() {
    super(...arguments);
    /**
     * The appearance of the element.
     *
     * @public
     * @remarks
     * HTML Attribute: appearance
     */

    this.appearance = "outline";
  }

}

__decorate([attr], NumberField.prototype, "appearance", void 0);
/**
 * A function that returns a {@link @microsoft/fast-foundation#NumberField} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#numberFieldTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-number-field>`
 *
 * {@link https://developer.mozilla.org/en-US/docs/Web/API/ShadowRoot/delegatesFocus | delegatesFocus}
 */


const fastNumberField = NumberField.compose({
  baseName: "number-field",
  baseClass: NumberField$1,
  styles: numberFieldStyles,
  template: numberFieldTemplate,
  shadowOptions: {
    delegatesFocus: true
  },
  stepDownGlyph:
  /* html */
  `
        <span class="step-down-glyph" part="step-down-glyph"></span>
    `,
  stepUpGlyph:
  /* html */
  `
        <span class="step-up-glyph" part="step-up-glyph"></span>
    `
});

/**
 * Styles for Picker
 * @public
 */

const pickerStyles = (context, definition) => css`
        .region{z-index:1000;overflow:hidden;display:flex;font-family:${bodyFont};font-size:${typeRampBaseFontSize}}.loaded{opacity:1;pointer-events:none}.loading-display,.no-options-display{background:${fillColor};width:100%;min-height:calc(${heightNumber} * 1px);display:flex;flex-direction:column;align-items:center;justify-items:center;padding:calc(${designUnit} * 1px)}.loading-progress{width:42px;height:42px}.bottom{flex-direction:column}.top{flex-direction:column-reverse}`;

/**
 * Styles for Picker menu
 * @public
 */

const pickerMenuStyles = (context, definition) => css`
        :host{background:${fillColor};--elevation:11;z-index:1000;display:flex;width:100%;max-height:100%;min-height:58px;box-sizing:border-box;flex-direction:column;overflow-y:auto;overflow-x:hidden;pointer-events:auto;border-radius:calc(${controlCornerRadius} * 1px);padding:calc(${designUnit} * 1px) 0;border:calc(${strokeWidth} * 1px) solid transparent;${elevation}}.suggestions-available-alert{height:0;opacity:0;overflow:hidden}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host{background:${SystemColors.Canvas};border-color:${SystemColors.CanvasText}}`));

/**
 * Styles for Picker menu option
 * @public
 */

const pickerMenuOptionStyles = (context, definition) => css`
        :host{display:flex;align-items:center;justify-items:center;font-family:${bodyFont};border-radius:calc(${controlCornerRadius} * 1px);border:calc(${focusStrokeWidth} * 1px) solid transparent;box-sizing:border-box;background:${neutralFillStealthRest};color:${neutralForegroundRest};cursor:pointer;fill:currentcolor;font-size:${typeRampBaseFontSize};min-height:calc(${heightNumber} * 1px);line-height:${typeRampBaseLineHeight};margin:0 calc(${designUnit} * 1px);outline:none;overflow:hidden;padding:0 calc(${designUnit} * 2.25px);user-select:none;white-space:nowrap}:host(:${focusVisible}[role="listitem"]){border-color:${focusStrokeOuter};background:${neutralFillStealthFocus}}:host(:hover){background:${neutralFillStealthHover}}:host(:active){background:${neutralFillStealthActive}}:host([aria-selected="true"]){background:${accentFillRest};color:${foregroundOnAccentRest}}:host([aria-selected="true"]:hover){background:${accentFillHover};color:${foregroundOnAccentHover}}:host([aria-selected="true"]:active){background:${accentFillActive};color:${foregroundOnAccentActive}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host{border-color:transparent;forced-color-adjust:none;color:${SystemColors.ButtonText};fill:currentcolor}:host(:not([aria-selected="true"]):hover),:host([aria-selected="true"]){background:${SystemColors.Highlight};color:${SystemColors.HighlightText}}:host([disabled]),:host([disabled]:not([aria-selected="true"]):hover){background:${SystemColors.Canvas};color:${SystemColors.GrayText};fill:currentcolor;opacity:1}`));

/**
 * Styles for Picker list
 * @public
 */

const pickerListStyles = (context, definition) => css`
        :host{display:flex;flex-direction:row;column-gap:calc(${designUnit} * 1px);row-gap:calc(${designUnit} * 1px);flex-wrap:wrap}::slotted([role="combobox"]){min-width:260px;width:auto;box-sizing:border-box;color:${neutralForegroundRest};background:${neutralFillInputRest};border-radius:calc(${controlCornerRadius} * 1px);border:calc(${strokeWidth} * 1px) solid ${accentFillRest};height:calc(${heightNumber} * 1px);font-family:${bodyFont};outline:none;user-select:none;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};padding:0 calc(${designUnit} * 2px + 1px)}::slotted([role="combobox"]:active){{background:${neutralFillInputHover};border-color:${accentFillActive}}::slotted([role="combobox"]:focus-within){border-color:${focusStrokeOuter};box-shadow:0 0 0 1px ${focusStrokeOuter} inset}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                ::slotted([role="combobox"]:active){background:${SystemColors.Field};border-color:${SystemColors.Highlight}}::slotted([role="combobox"]:focus-within){border-color:${SystemColors.Highlight};box-shadow:0 0 0 1px ${SystemColors.Highlight} inset}::slotted(input:placeholder){color:${SystemColors.GrayText}}`));

/**
 * Styles for Picker list item
 * @public
 */

const pickerListItemStyles = (context, definition) => css`
        :host{display:flex;align-items:center;justify-items:center;font-family:${bodyFont};border-radius:calc(${controlCornerRadius} * 1px);border:calc(${focusStrokeWidth} * 1px) solid transparent;box-sizing:border-box;background:${neutralFillStealthRest};color:${neutralForegroundRest};cursor:pointer;fill:currentcolor;font-size:${typeRampBaseFontSize};height:calc(${heightNumber} * 1px);line-height:${typeRampBaseLineHeight};outline:none;overflow:hidden;padding:0 calc(${designUnit} * 2.25px);user-select:none;white-space:nowrap}:host(:hover){background:${neutralFillStealthHover}}:host(:active){background:${neutralFillStealthActive}}:host(:${focusVisible}){background:${neutralFillStealthFocus};border-color:${focusStrokeOuter}}:host([aria-selected="true"]){background:${accentFillRest};color:${foregroundOnAccentActive}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host{border-color:transparent;forced-color-adjust:none;color:${SystemColors.ButtonText};fill:currentcolor}:host(:not([aria-selected="true"]):hover),:host([aria-selected="true"]){background:${SystemColors.Highlight};color:${SystemColors.HighlightText}}:host([disabled]),:host([disabled]:not([aria-selected="true"]):hover){background:${SystemColors.Canvas};color:${SystemColors.GrayText};fill:currentcolor;opacity:1}`));

/**
 * The FAST  Picker Custom Element. Implements {@link @microsoft/fast-foundation#Picker},
 * {@link @microsoft/fast-foundation#PickerTemplate}
 *
 *
 * @alpha
 * @remarks
 * * Generates HTML Element: `<fast-picker>`
 */

const fastPicker = Picker.compose({
  baseName: "picker",
  template: pickerTemplate,
  styles: pickerStyles,
  shadowOptions: {}
});
/**
 * @public
 */

class PickerMenu extends PickerMenu$1 {
  /**
   * @public
   */
  connectedCallback() {
    fillColor.setValueFor(this, neutralLayerFloating);
    super.connectedCallback();
  }

}
/**
 * Component that displays the list of available picker options
 *
 *
 * @alpha
 * @remarks
 * HTML Element: \<fast-picker-menu\>
 */

const fastPickerMenu = PickerMenu.compose({
  baseName: "picker-menu",
  baseClass: PickerMenu$1,
  template: pickerMenuTemplate,
  styles: pickerMenuStyles
});
/**
 *  Component that displays available picker menu options
 *
 *
 * @alpha
 * @remarks
 * HTML Element: \<fast-picker-menu-option\>
 */

const fastPickerMenuOption = PickerMenuOption.compose({
  baseName: "picker-menu-option",
  template: pickerMenuOptionTemplate,
  styles: pickerMenuOptionStyles
});
/**
 * Component that displays the list of selected picker items along
 * with the input combobox
 *
 * @alpha
 * @remarks
 * HTML Element: \<fast-picker-list\>
 *
 */

const fastPickerList = PickerList.compose({
  baseName: "picker-list",
  template: pickerListTemplate,
  styles: pickerListStyles
});
/**
 * Component that displays selected items
 *
 * @alpha
 * @remarks
 * HTML Element: \<fast-picker-list-item\>
 */

const fastPickerListItem = PickerListItem.compose({
  baseName: "picker-list-item",
  template: pickerListItemTemplate,
  styles: pickerListItemStyles
});

/**
 * Styles for Progress Ring
 * @public
 */

const progressRingStyles = (context, definition) => css`
        ${display("flex")} :host{align-items:center;outline:none;height:calc(${heightNumber} * 1px);width:calc(${heightNumber} * 1px);margin:calc(${heightNumber} * 1px) 0}.progress{height:100%;width:100%}.background{stroke:${neutralFillRest};fill:none;stroke-width:2px}.determinate{stroke:${accentForegroundRest};fill:none;stroke-width:2px;stroke-linecap:round;transform-origin:50% 50%;transform:rotate(-90deg);transition:all 0.2s ease-in-out}.indeterminate-indicator-1{stroke:${accentForegroundRest};fill:none;stroke-width:2px;stroke-linecap:round;transform-origin:50% 50%;transform:rotate(-90deg);transition:all 0.2s ease-in-out;animation:spin-infinite 2s linear infinite}:host([paused]) .indeterminate-indicator-1{animation-play-state:paused;stroke:${neutralFillRest}}:host([paused]) .determinate{stroke:${neutralForegroundHint}}@keyframes spin-infinite{0%{stroke-dasharray:0.01px 43.97px;transform:rotate(0deg)}50%{stroke-dasharray:21.99px 21.99px;transform:rotate(450deg)}100%{stroke-dasharray:0.01px 43.97px;transform:rotate(1080deg)}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                .indeterminate-indicator-1,.determinate{stroke:${SystemColors.FieldText}}.background{stroke:${SystemColors.Field}}:host([paused]) .indeterminate-indicator-1{stroke:${SystemColors.Field}}:host([paused]) .determinate{stroke:${SystemColors.GrayText}}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#BaseProgress} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#progressRingTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-progress-ring>`
 */

const fastProgressRing = BaseProgress.compose({
  baseName: "progress-ring",
  template: progressRingTemplate,
  styles: progressRingStyles,
  indeterminateIndicator:
  /* html */
  `
        <svg class="progress" part="progress" viewBox="0 0 16 16">
            <circle
                class="background"
                part="background"
                cx="8px"
                cy="8px"
                r="7px"
            ></circle>
            <circle
                class="indeterminate-indicator-1"
                part="indeterminate-indicator-1"
                cx="8px"
                cy="8px"
                r="7px"
            ></circle>
        </svg>
    `
});

/**
 * Styles for Progress
 * @public
 */

const progressStyles = (context, definition) => css`
        ${display("flex")} :host{align-items:center;outline:none;height:calc(${designUnit} * 1px);margin:calc(${designUnit} * 1px) 0}.progress{background-color:${neutralFillRest};border-radius:calc(${designUnit} * 1px);width:100%;height:100%;display:flex;align-items:center;position:relative}.determinate{background-color:${accentForegroundRest};border-radius:calc(${designUnit} * 1px);height:100%;transition:all 0.2s ease-in-out;display:flex}.indeterminate{height:100%;border-radius:calc(${designUnit} * 1px);display:flex;width:100%;position:relative;overflow:hidden}.indeterminate-indicator-1{position:absolute;opacity:0;height:100%;background-color:${accentForegroundRest};border-radius:calc(${designUnit} * 1px);animation-timing-function:cubic-bezier(0.4,0,0.6,1);width:40%;animation:indeterminate-1 2s infinite}.indeterminate-indicator-2{position:absolute;opacity:0;height:100%;background-color:${accentForegroundRest};border-radius:calc(${designUnit} * 1px);animation-timing-function:cubic-bezier(0.4,0,0.6,1);width:60%;animation:indeterminate-2 2s infinite}:host([paused]) .indeterminate-indicator-1,:host([paused]) .indeterminate-indicator-2{animation-play-state:paused;background-color:${neutralFillRest}}:host([paused]) .determinate{background-color:${neutralForegroundHint}}@keyframes indeterminate-1{0%{opacity:1;transform:translateX(-100%)}70%{opacity:1;transform:translateX(300%)}70.01%{opacity:0}100%{opacity:0;transform:translateX(300%)}}@keyframes indeterminate-2{0%{opacity:0;transform:translateX(-150%)}29.99%{opacity:0}30%{opacity:1;transform:translateX(-150%)}100%{transform:translateX(166.66%);opacity:1}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                .progress{forced-color-adjust:none;background-color:${SystemColors.Field};box-shadow:0 0 0 1px inset ${SystemColors.FieldText}}.determinate,.indeterminate-indicator-1,.indeterminate-indicator-2{forced-color-adjust:none;background-color:${SystemColors.FieldText}}:host([paused]) .determinate,:host([paused]) .indeterminate-indicator-1,:host([paused]) .indeterminate-indicator-2{background-color:${SystemColors.GrayText}}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#BaseProgress} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#progressTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-progress>`
 */

const fastProgress = BaseProgress.compose({
  baseName: "progress",
  template: progressTemplate,
  styles: progressStyles,
  indeterminateIndicator1:
  /* html */
  `
        <span class="indeterminate-indicator-1" part="indeterminate-indicator-1"></span>
    `,
  indeterminateIndicator2:
  /* html */
  `
        <span class="indeterminate-indicator-1" part="indeterminate-indicator-1"></span>
    `
});

/**
 * Styles for Radio Group
 * @public
 */

const radioGroupStyles = (context, definition) => css`
    ${display("flex")} :host{align-items:flex-start;margin:calc(${designUnit} * 1px) 0;flex-direction:column}.positioning-region{display:flex;flex-wrap:wrap}:host([orientation="vertical"]) .positioning-region{flex-direction:column}:host([orientation="horizontal"]) .positioning-region{flex-direction:row}`;

/**
 * A function that returns a {@link @microsoft/fast-foundation#RadioGroup} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#radioGroupTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-radio-group>`
 */

const fastRadioGroup = RadioGroup.compose({
  baseName: "radio-group",
  template: radioGroupTemplate,
  styles: radioGroupStyles
});

/**
 * Styles for Radio
 * @public
 */

const radioStyles = (context, definition) => css`
        ${display("inline-flex")} :host{--input-size:calc((${heightNumber} / 2) + ${designUnit});align-items:center;outline:none;margin:calc(${designUnit} * 1px) 0;user-select:none;position:relative;flex-direction:row;transition:all 0.2s ease-in-out}.control{position:relative;width:calc((${heightNumber} / 2 + ${designUnit}) * 1px);height:calc((${heightNumber} / 2 + ${designUnit}) * 1px);box-sizing:border-box;border-radius:999px;border:calc(${strokeWidth} * 1px) solid ${neutralStrokeRest};background:${neutralFillInputRest};outline:none;cursor:pointer}.label{font-family:${bodyFont};color:${neutralForegroundRest};padding-inline-start:calc(${designUnit} * 2px + 2px);margin-inline-end:calc(${designUnit} * 2px + 2px);cursor:pointer;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight}}.label__hidden{display:none;visibility:hidden}.control,.checked-indicator{flex-shrink:0}.checked-indicator{position:absolute;top:5px;left:5px;right:5px;bottom:5px;border-radius:999px;display:inline-block;background:${foregroundOnAccentRest};fill:${foregroundOnAccentRest};opacity:0;pointer-events:none}:host(:not([disabled])) .control:hover{background:${neutralFillInputHover};border-color:${neutralStrokeHover}}:host(:not([disabled])) .control:active{background:${neutralFillInputActive};border-color:${neutralStrokeActive}}:host(:${focusVisible}) .control{box-shadow:0 0 0 2px ${fillColor},0 0 0 4px ${focusStrokeOuter}}:host([aria-checked="true"]) .control{background:${accentFillRest};border:calc(${strokeWidth} * 1px) solid ${accentFillRest}}:host([aria-checked="true"]:not([disabled])) .control:hover{background:${accentFillHover};border:calc(${strokeWidth} * 1px) solid ${accentFillHover}}:host([aria-checked="true"]:not([disabled])) .control:hover .checked-indicator{background:${foregroundOnAccentHover};fill:${foregroundOnAccentHover}}:host([aria-checked="true"]:not([disabled])) .control:active{background:${accentFillActive};border:calc(${strokeWidth} * 1px) solid ${accentFillActive}}:host([aria-checked="true"]:not([disabled])) .control:active .checked-indicator{background:${foregroundOnAccentActive};fill:${foregroundOnAccentActive}}:host([aria-checked="true"]:${focusVisible}:not([disabled])) .control{box-shadow:0 0 0 2px ${fillColor},0 0 0 4px ${focusStrokeOuter}}:host([disabled]) .label,:host([readonly]) .label,:host([readonly]) .control,:host([disabled]) .control{cursor:${disabledCursor}}:host([aria-checked="true"]) .checked-indicator{opacity:1}:host([disabled]){opacity:${disabledOpacity}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            .control,:host([aria-checked="true"]:not([disabled])) .control{forced-color-adjust:none;border-color:${SystemColors.FieldText};background:${SystemColors.Field}}:host(:not([disabled])) .control:hover{border-color:${SystemColors.Highlight};background:${SystemColors.Field}}:host([aria-checked="true"]:not([disabled])) .control:hover,:host([aria-checked="true"]:not([disabled])) .control:active{border-color:${SystemColors.Highlight};background:${SystemColors.Highlight}}:host([aria-checked="true"]) .checked-indicator{background:${SystemColors.Highlight};fill:${SystemColors.Highlight}}:host([aria-checked="true"]:not([disabled])) .control:hover .checked-indicator,:host([aria-checked="true"]:not([disabled])) .control:active .checked-indicator{background:${SystemColors.HighlightText};fill:${SystemColors.HighlightText}}:host(:${focusVisible}) .control{border-color:${SystemColors.Highlight};box-shadow:0 0 0 2px ${SystemColors.Field},0 0 0 4px ${SystemColors.FieldText}}:host([aria-checked="true"]:${focusVisible}:not([disabled])) .control{border-color:${SystemColors.Highlight};box-shadow:0 0 0 2px ${SystemColors.Field},0 0 0 4px ${SystemColors.FieldText}}:host([disabled]){forced-color-adjust:none;opacity:1}:host([disabled]) .label{color:${SystemColors.GrayText}}:host([disabled]) .control,:host([aria-checked="true"][disabled]) .control:hover,.control:active{background:${SystemColors.Field};border-color:${SystemColors.GrayText}}:host([disabled]) .checked-indicator,:host([aria-checked="true"][disabled]) .control:hover .checked-indicator{fill:${SystemColors.GrayText};background:${SystemColors.GrayText}}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#Radio} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#radioTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-radio>`
 */

const fastRadio = Radio.compose({
  baseName: "radio",
  template: radioTemplate,
  styles: radioStyles,
  checkedIndicator:
  /* html */
  `
        <div part="checked-indicator" class="checked-indicator"></div>
    `
});

const clearButtonHover = DesignToken.create("clear-button-hover").withDefault(target => {
  const buttonRecipe = neutralFillStealthRecipe.getValueFor(target);
  const inputRecipe = neutralFillRecipe.getValueFor(target);
  return buttonRecipe.evaluate(target, inputRecipe.evaluate(target).hover).hover;
});
const clearButtonActive = DesignToken.create("clear-button-active").withDefault(target => {
  const buttonRecipe = neutralFillStealthRecipe.getValueFor(target);
  const inputRecipe = neutralFillRecipe.getValueFor(target);
  return buttonRecipe.evaluate(target, inputRecipe.evaluate(target).hover).active;
});
const searchStyles$1 = (context, definition) => css`
    ${display("inline-block")} :host{font-family:${bodyFont};outline:none;user-select:none}.root{box-sizing:border-box;position:relative;display:flex;flex-direction:row;color:${neutralForegroundRest};background:${neutralFillInputRest};border-radius:calc(${controlCornerRadius} * 1px);border:calc(${strokeWidth} * 1px) solid ${accentFillRest};height:calc(${heightNumber} * 1px);align-items:baseline}.control{-webkit-appearance:none;font:inherit;background:transparent;border:0;color:inherit;height:calc(100% - 4px);width:100%;margin-top:auto;margin-bottom:auto;border:none;padding:0 calc(${designUnit} * 2px + 1px);font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight}}.control::-webkit-search-cancel-button{-webkit-appearance:none}.control:hover,.control:${focusVisible},.control:disabled,.control:active{outline:none}.clear-button{height:calc(100% - 2px);opacity:0;margin:1px;background:transparent;color:${neutralForegroundRest};fill:currentcolor;border:none;border-radius:calc(${controlCornerRadius} * 1px);min-width:calc(${heightNumber} * 1px);font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};outline:none;font-family:${bodyFont};padding:0 calc((10 + (${designUnit} * 2 * ${density})) * 1px)}.clear-button:hover{background:${neutralFillStealthHover}}.clear-button:active{background:${neutralFillStealthActive}}:host([appearance="filled"]) .clear-button:hover{background:${clearButtonHover}}:host([appearance="filled"]) .clear-button:active{background:${clearButtonActive}}.input-wrapper{display:flex;position:relative;width:100%;height:100%}.label{display:block;color:${neutralForegroundRest};cursor:pointer;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};margin-bottom:4px}.label__hidden{display:none;visibility:hidden}.input-wrapper,.start,.end{align-self:center}.start,.end{display:flex;margin:1px;fill:currentcolor}::slotted([slot="end"]){height:100%}.end{margin-inline-end:1px;height:calc(100% - 2px)}::slotted(svg){width:16px;height:16px;margin-inline-end:11px;margin-inline-start:11px;margin-top:auto;margin-bottom:auto}:host(:hover:not([disabled])) .root{background:${neutralFillInputHover};border-color:${accentFillHover}}:host(:active:not([disabled])) .root{background:${neutralFillInputHover};border-color:${accentFillActive}}:host(:focus-within:not([disabled])) .root{border-color:${focusStrokeOuter};box-shadow:0 0 0 1px ${focusStrokeOuter} inset}.clear-button__hidden{opacity:0}:host(:hover:not([disabled],[readOnly])) .clear-button,:host(:active:not([disabled],[readOnly])) .clear-button,:host(:focus-within:not([disabled],[readOnly])) .clear-button{opacity:1}:host(:hover:not([disabled],[readOnly])) .clear-button__hidden,:host(:active:not([disabled],[readOnly])) .clear-button__hidden,:host(:focus-within:not([disabled],[readOnly])) .clear-button__hidden{opacity:0}:host([appearance="filled"]) .root{background:${fillColor}}:host([appearance="filled"]:hover:not([disabled])) .root{background:${neutralFillHover}}:host([disabled]) .label,:host([readonly]) .label,:host([readonly]) .control,:host([disabled]) .control{cursor:${disabledCursor}}:host([disabled]){opacity:${disabledOpacity}}:host([disabled]) .control{border-color:${neutralStrokeRest}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                .root,:host([appearance="filled"]) .root{forced-color-adjust:none;background:${SystemColors.Field};border-color:${SystemColors.FieldText}}:host(:hover:not([disabled])) .root,:host([appearance="filled"]:hover:not([disabled])) .root,:host([appearance="filled"]:hover) .root{background:${SystemColors.Field};border-color:${SystemColors.Highlight}}.start,.end{fill:currentcolor}:host([disabled]){opacity:1}:host([disabled]) .root,:host([appearance="filled"]:hover[disabled]) .root{border-color:${SystemColors.GrayText};background:${SystemColors.Field}}:host(:focus-within:enabled) .root{border-color:${SystemColors.Highlight};box-shadow:0 0 0 1px ${SystemColors.Highlight} inset}input::placeholder{color:${SystemColors.GrayText}}`));

/**
 * @internal
 */

class Search extends Search$1 {
  constructor() {
    super(...arguments);
    /**
     * The appearance of the element.
     *
     * @public
     * @remarks
     * HTML Attribute: appearance
     */

    this.appearance = "outline";
  }

}

__decorate([attr], Search.prototype, "appearance", void 0);
/**
 * A function that returns a {@link @microsoft/fast-foundation#Search} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#searchTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: \<fast-search\>
 *
 * {@link https://developer.mozilla.org/en-US/docs/Web/API/ShadowRoot/delegatesFocus | delegatesFocus}
 */


const fastSearch = Search.compose({
  baseName: "search",
  baseClass: Search$1,
  template: searchTemplate,
  styles: searchStyles$1,
  shadowOptions: {
    delegatesFocus: true
  }
});
/**
 * Styles for Search
 * @public
 */

const searchStyles = searchStyles$1;

/**
 * Base class for Select.
 * @public
 */

class Select extends Select$1 {
  constructor() {
    super(...arguments);
    /**
     * The cached scroll width of the listbox when visible.
     *
     * @internal
     */

    this.listboxScrollWidth = "";
  }
  /**
   * @internal
   */


  connectedCallback() {
    super.connectedCallback();

    if (this.listbox) {
      fillColor.setValueFor(this.listbox, neutralLayerFloating);
    }
  }
  /**
   * Returns the calculated max height for the listbox.
   *
   * @internal
   * @remarks
   * Used to generate the `--listbox-max-height` CSS custom property.
   *
   */


  get listboxMaxHeight() {
    return Math.floor(this.maxHeight / heightNumberAsToken.getValueFor(this)).toString();
  }
  /**
   * @internal
   */


  listboxScrollWidthChanged() {
    this.updateComputedStylesheet();
  }
  /**
   * Returns the size value, if any. Otherwise, returns 4 if in
   * multi-selection mode, or 0 if in single-selection mode.
   *
   * @internal
   * @remarks
   * Used to generate the `--size` CSS custom property.
   *
   */


  get selectSize() {
    var _a;

    return `${(_a = this.size) !== null && _a !== void 0 ? _a : this.multiple ? 4 : 0}`;
  }
  /**
   * Updates the computed stylesheet when the multiple property changes.
   *
   * @param prev - the previous multiple value
   * @param next - the current multiple value
   *
   * @override
   * @internal
   */


  multipleChanged(prev, next) {
    super.multipleChanged(prev, next);
    this.updateComputedStylesheet();
  }
  /**
   * Sets the selectMaxSize design token when the maxHeight property changes.
   *
   * @param prev - the previous maxHeight value
   * @param next - the current maxHeight value
   *
   * @internal
   */


  maxHeightChanged(prev, next) {
    if (this.collapsible) {
      this.updateComputedStylesheet();
    }
  }

  setPositioning() {
    super.setPositioning();
    this.updateComputedStylesheet();
  }
  /**
   * Updates the component dimensions when the size property is changed.
   *
   * @param prev - the previous size value
   * @param next - the current size value
   *
   * @override
   * @internal
   */


  sizeChanged(prev, next) {
    super.sizeChanged(prev, next);
    this.updateComputedStylesheet();

    if (this.collapsible) {
      requestAnimationFrame(() => {
        this.listbox.style.setProperty("display", "flex");
        this.listbox.style.setProperty("overflow", "visible");
        this.listbox.style.setProperty("visibility", "hidden");
        this.listbox.style.setProperty("width", "auto");
        this.listbox.hidden = false;
        this.listboxScrollWidth = `${this.listbox.scrollWidth}`;
        this.listbox.hidden = true;
        this.listbox.style.removeProperty("display");
        this.listbox.style.removeProperty("overflow");
        this.listbox.style.removeProperty("visibility");
        this.listbox.style.removeProperty("width");
      });
      return;
    }

    this.listboxScrollWidth = "";
  }
  /**
   * Updates an internal stylesheet with calculated CSS custom properties.
   *
   * @internal
   */


  updateComputedStylesheet() {
    if (this.computedStylesheet) {
      this.$fastController.removeStyles(this.computedStylesheet);
    }

    this.computedStylesheet = css`
            :host{--listbox-max-height:${this.listboxMaxHeight};--listbox-scroll-width:${this.listboxScrollWidth};--size:${this.selectSize}}`;
    this.$fastController.addStyles(this.computedStylesheet);
  }

}

__decorate([observable], Select.prototype, "listboxScrollWidth", void 0);
/**
 * A function that returns a {@link @microsoft/fast-foundation#Select} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#selectTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-select>`
 *
 */


const fastSelect = Select.compose({
  baseName: "select",
  baseClass: Select$1,
  template: selectTemplate,
  styles: selectStyles,
  indicator:
  /* html */
  `
        <svg
            class="select-indicator"
            part="select-indicator"
            viewBox="0 0 12 7"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                d="M11.85.65c.2.2.2.5 0 .7L6.4 6.84a.55.55 0 01-.78 0L.14 1.35a.5.5 0 11.71-.7L6 5.8 11.15.65c.2-.2.5-.2.7 0z"
            />
        </svg>
    `
});

/**
 * Styles for Skeleton
 * @public
 */

const skeletonStyles = (context, definition) => css`
        ${display("block")} :host{--skeleton-fill-default:#e1dfdd;overflow:hidden;width:100%;position:relative;background-color:var(--skeleton-fill,var(--skeleton-fill-default));--skeleton-animation-gradient-default:linear-gradient(
                270deg,var(--skeleton-fill,var(--skeleton-fill-default)) 0%,#f3f2f1 51.13%,var(--skeleton-fill,var(--skeleton-fill-default)) 100%
            );--skeleton-animation-timing-default:ease-in-out}:host([shape="rect"]){border-radius:calc(${controlCornerRadius} * 1px)}:host([shape="circle"]){border-radius:100%;overflow:hidden}object{position:absolute;width:100%;height:auto;z-index:2}object img{width:100%;height:auto}${display("block")} span.shimmer{position:absolute;width:100%;height:100%;background-image:var(
                --skeleton-animation-gradient,var(--skeleton-animation-gradient-default)
            );background-size:0px 0px / 90% 100%;background-repeat:no-repeat;background-color:var(--skeleton-animation-fill,${neutralFillRest});animation:shimmer 2s infinite;animation-timing-function:var(
                --skeleton-animation-timing,var(--skeleton-timing-default)
            );animation-direction:normal;z-index:1}::slotted(svg){z-index:2}::slotted(.pattern){width:100%;height:100%}@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host{forced-color-adjust:none;background-color:${SystemColors.ButtonFace};box-shadow:0 0 0 1px ${SystemColors.ButtonText}}${display("block")} span.shimmer{display:none}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#Skeleton} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#skeletonTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-skeleton>`
 */

const fastSkeleton = Skeleton.compose({
  baseName: "skeleton",
  template: skeletonTemplate,
  styles: skeletonStyles
});

/**
 * Styles for Horizontal Slider label
 * @public
 */

const horizontalSliderLabelStyles = css`
    :host{align-self:start;grid-row:2;margin-top:-2px;height:calc((${heightNumber} / 2 + ${designUnit}) * 1px);width:auto}.container{grid-template-rows:auto auto;grid-template-columns:0}.label{margin:2px 0}`;
/**
 * Styles for Vertical slider label
 * @public
 */

const verticalSliderLabelStyles = css`
    :host{justify-self:start;grid-column:2;margin-left:2px;height:auto;width:calc((${heightNumber} / 2 + ${designUnit}) * 1px)}.container{grid-template-columns:auto auto;grid-template-rows:0;min-width:calc(var(--thumb-size) * 1px);height:calc(var(--thumb-size) * 1px)}.mark{transform:rotate(90deg);align-self:center}.label{margin-left:calc((${designUnit} / 2) * 3px);align-self:center}`;
/**
 * Styles for Slider Label
 * @public
 */

const sliderLabelStyles = (context, definition) => css`
        ${display("block")} :host{font-family:${bodyFont};color:${neutralForegroundRest};fill:currentcolor}.root{position:absolute;display:grid}.container{display:grid;justify-self:center}.label{justify-self:center;align-self:center;white-space:nowrap;max-width:30px}.mark{width:calc((${designUnit} / 4) * 1px);height:calc(${heightNumber} * 0.25 * 1px);background:${neutralStrokeRest};justify-self:center}:host(.disabled){opacity:${disabledOpacity}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                .mark{forced-color-adjust:none;background:${SystemColors.FieldText}}:host(.disabled){forced-color-adjust:none;opacity:1}:host(.disabled) .label{color:${SystemColors.GrayText}}:host(.disabled) .mark{background:${SystemColors.GrayText}}`));

/**
 * @internal
 */

class SliderLabel extends SliderLabel$1 {
  sliderOrientationChanged() {
    if (this.sliderOrientation === Orientation.horizontal) {
      this.$fastController.addStyles(horizontalSliderLabelStyles);
      this.$fastController.removeStyles(verticalSliderLabelStyles);
    } else {
      this.$fastController.addStyles(verticalSliderLabelStyles);
      this.$fastController.removeStyles(horizontalSliderLabelStyles);
    }
  }

}
/**
 * A function that returns a {@link @microsoft/fast-foundation#SliderLabel} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#sliderLabelTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-slider-label>`
 */

const fastSliderLabel = SliderLabel.compose({
  baseName: "slider-label",
  baseClass: SliderLabel$1,
  template: sliderLabelTemplate,
  styles: sliderLabelStyles
});

const ltr$1 = css`
    .track-start{left:0}`;
const rtl$1 = css`
    .track-start{right:0}`;
/**
 * Styles for Slider
 * @public
 */

const sliderStyles = (context, definition) => css`
        :host([hidden]){display:none}${display("inline-grid")} :host{--thumb-size:calc(${heightNumber} * 0.5 - ${designUnit});--thumb-translate:calc(var(--thumb-size) * -0.5 + var(--track-width) / 2);--track-overhang:calc((${designUnit} / 2) * -1);--track-width:${designUnit};--fast-slider-height:calc(var(--thumb-size) * 10);align-items:center;width:100%;margin:calc(${designUnit} * 1px) 0;user-select:none;box-sizing:border-box;border-radius:calc(${controlCornerRadius} * 1px);outline:none;cursor:pointer}:host([orientation="horizontal"]) .positioning-region{position:relative;margin:0 8px;display:grid;grid-template-rows:calc(var(--thumb-size) * 1px) 1fr}:host([orientation="vertical"]) .positioning-region{position:relative;margin:0 8px;display:grid;height:100%;grid-template-columns:calc(var(--thumb-size) * 1px) 1fr}:host(:${focusVisible}) .thumb-cursor{box-shadow:0 0 0 2px ${fillColor},0 0 0 4px ${focusStrokeOuter}}.thumb-container{position:absolute;height:calc(var(--thumb-size) * 1px);width:calc(var(--thumb-size) * 1px);transition:all 0.2s ease;color:${neutralForegroundRest};fill:currentcolor}.thumb-cursor{border:none;width:calc(var(--thumb-size) * 1px);height:calc(var(--thumb-size) * 1px);background:${neutralForegroundRest};border-radius:calc(${controlCornerRadius} * 1px)}.thumb-cursor:hover{background:${neutralForegroundRest};border-color:${neutralStrokeHover}}.thumb-cursor:active{background:${neutralForegroundRest}}.track-start{background:${accentForegroundRest};position:absolute;height:100%;left:0;border-radius:calc(${controlCornerRadius} * 1px)}:host([orientation="horizontal"]) .thumb-container{transform:translateX(calc(var(--thumb-size) * 0.5px)) translateY(calc(var(--thumb-translate) * 1px))}:host([orientation="vertical"]) .thumb-container{transform:translateX(calc(var(--thumb-translate) * 1px)) translateY(calc(var(--thumb-size) * 0.5px))}:host([orientation="horizontal"]){min-width:calc(var(--thumb-size) * 1px)}:host([orientation="horizontal"]) .track{right:calc(var(--track-overhang) * 1px);left:calc(var(--track-overhang) * 1px);align-self:start;height:calc(var(--track-width) * 1px)}:host([orientation="vertical"]) .track{top:calc(var(--track-overhang) * 1px);bottom:calc(var(--track-overhang) * 1px);width:calc(var(--track-width) * 1px);height:100%}.track{background:${neutralStrokeRest};position:absolute;border-radius:calc(${controlCornerRadius} * 1px)}:host([orientation="vertical"]){height:calc(var(--fast-slider-height) * 1px);min-height:calc(var(--thumb-size) * 1px);min-width:calc(${designUnit} * 20px)}:host([orientation="vertical"]) .track-start{height:auto;width:100%;top:0}:host([disabled]),:host([readonly]){cursor:${disabledCursor}}:host([disabled]){opacity:${disabledOpacity}}`.withBehaviors(new DirectionalStyleSheetBehavior(ltr$1, rtl$1), forcedColorsStylesheetBehavior(css`
                .thumb-cursor{forced-color-adjust:none;border-color:${SystemColors.FieldText};background:${SystemColors.FieldText}}.thumb-cursor:hover,.thumb-cursor:active{background:${SystemColors.Highlight}}.track{forced-color-adjust:none;background:${SystemColors.FieldText}}:host(:${focusVisible}) .thumb-cursor{border-color:${SystemColors.Highlight}}:host([disabled]){opacity:1}:host([disabled]) .track,:host([disabled]) .thumb-cursor{forced-color-adjust:none;background:${SystemColors.GrayText}}:host(:${focusVisible}) .thumb-cursor{background:${SystemColors.Highlight};border-color:${SystemColors.Highlight};box-shadow:0 0 0 2px ${SystemColors.Field},0 0 0 4px ${SystemColors.FieldText}}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#Slider} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#sliderTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-slider>`
 */

const fastSlider = Slider.compose({
  baseName: "slider",
  template: sliderTemplate,
  styles: sliderStyles,
  thumb:
  /* html */
  `
        <div class="thumb-cursor"></div>
    `
});

/**
 * Styles for Switch
 * @public
 */

const switchStyles = (context, definition) => css`
        :host([hidden]){display:none}${display("inline-flex")} :host{align-items:center;outline:none;font-family:${bodyFont};margin:calc(${designUnit} * 1px) 0;${
/*
 * Chromium likes to select label text or the default slot when
 * the checkbox is clicked. Maybe there is a better solution here?
 */
""} user-select:none}:host([disabled]){opacity:${disabledOpacity}}:host([disabled]) .label,:host([readonly]) .label,:host([readonly]) .switch,:host([disabled]) .switch{cursor:${disabledCursor}}.switch{position:relative;outline:none;box-sizing:border-box;width:calc(${heightNumber} * 1px);height:calc((${heightNumber} / 2 + ${designUnit}) * 1px);background:${neutralFillInputRest};border-radius:calc(${controlCornerRadius} * 1px);border:calc(${strokeWidth} * 1px) solid ${neutralStrokeRest}}.switch:hover{background:${neutralFillInputHover};border-color:${neutralStrokeHover};cursor:pointer}host([disabled]) .switch:hover,host([readonly]) .switch:hover{background:${neutralFillInputHover};border-color:${neutralStrokeHover};cursor:${disabledCursor}}:host(:not([disabled])) .switch:active{background:${neutralFillInputActive};border-color:${neutralStrokeActive}}:host(:${focusVisible}) .switch{box-shadow:0 0 0 2px ${fillColor},0 0 0 4px ${focusStrokeOuter}}.checked-indicator{position:absolute;top:5px;bottom:5px;background:${neutralForegroundRest};border-radius:calc(${controlCornerRadius} * 1px);transition:all 0.2s ease-in-out}.status-message{color:${neutralForegroundRest};cursor:pointer;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight}}:host([disabled]) .status-message,:host([readonly]) .status-message{cursor:${disabledCursor}}.label{color:${neutralForegroundRest};margin-inline-end:calc(${designUnit} * 2px + 2px);font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};cursor:pointer}.label__hidden{display:none;visibility:hidden}::slotted([slot="checked-message"]),::slotted([slot="unchecked-message"]){margin-inline-start:calc(${designUnit} * 2px + 2px)}:host([aria-checked="true"]) .checked-indicator{background:${foregroundOnAccentRest}}:host([aria-checked="true"]) .switch{background:${accentFillRest};border-color:${accentFillRest}}:host([aria-checked="true"]:not([disabled])) .switch:hover{background:${accentFillHover};border-color:${accentFillHover}}:host([aria-checked="true"]:not([disabled])) .switch:hover .checked-indicator{background:${foregroundOnAccentHover}}:host([aria-checked="true"]:not([disabled])) .switch:active{background:${accentFillActive};border-color:${accentFillActive}}:host([aria-checked="true"]:not([disabled])) .switch:active .checked-indicator{background:${foregroundOnAccentActive}}:host([aria-checked="true"]:${focusVisible}:not([disabled])) .switch{box-shadow:0 0 0 2px ${fillColor},0 0 0 4px ${focusStrokeOuter}}.unchecked-message{display:block}.checked-message{display:none}:host([aria-checked="true"]) .unchecked-message{display:none}:host([aria-checked="true"]) .checked-message{display:block}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            .checked-indicator,:host(:not([disabled])) .switch:active .checked-indicator{forced-color-adjust:none;background:${SystemColors.FieldText}}.switch{forced-color-adjust:none;background:${SystemColors.Field};border-color:${SystemColors.FieldText}}:host(:not([disabled])) .switch:hover{background:${SystemColors.HighlightText};border-color:${SystemColors.Highlight}}:host([aria-checked="true"]) .switch{background:${SystemColors.Highlight};border-color:${SystemColors.Highlight}}:host([aria-checked="true"]:not([disabled])) .switch:hover,:host(:not([disabled])) .switch:active{background:${SystemColors.HighlightText};border-color:${SystemColors.Highlight}}:host([aria-checked="true"]) .checked-indicator{background:${SystemColors.HighlightText}}:host([aria-checked="true"]:not([disabled])) .switch:hover .checked-indicator{background:${SystemColors.Highlight}}:host([disabled]){opacity:1}:host(:${focusVisible}) .switch{border-color:${SystemColors.Highlight};box-shadow:0 0 0 2px ${SystemColors.Field},0 0 0 4px ${SystemColors.FieldText}}:host([aria-checked="true"]:${focusVisible}:not([disabled])) .switch{box-shadow:0 0 0 2px ${SystemColors.Field},0 0 0 4px ${SystemColors.FieldText}}:host([disabled]) .checked-indicator{background:${SystemColors.GrayText}}:host([disabled]) .switch{background:${SystemColors.Field};border-color:${SystemColors.GrayText}}`), new DirectionalStyleSheetBehavior(css`
                .checked-indicator{left:5px;right:calc(((${heightNumber} / 2) + 1) * 1px)}:host([aria-checked="true"]) .checked-indicator{left:calc(((${heightNumber} / 2) + 1) * 1px);right:5px}`, css`
                .checked-indicator{right:5px;left:calc(((${heightNumber} / 2) + 1) * 1px)}:host([aria-checked="true"]) .checked-indicator{right:calc(((${heightNumber} / 2) + 1) * 1px);left:5px}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#Switch} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#switchTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-switch>`
 */

const fastSwitch = Switch.compose({
  baseName: "switch",
  template: switchTemplate,
  styles: switchStyles,
  switch:
  /* html */
  `
        <span class="checked-indicator" part="checked-indicator"></span>
    `
});

/**
 * Styles for Tabs
 * @public
 */

const tabsStyles = (context, definition) => css`
        ${display("grid")} :host{box-sizing:border-box;font-family:${bodyFont};font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};color:${neutralForegroundRest};grid-template-columns:auto 1fr auto;grid-template-rows:auto 1fr}.tablist{display:grid;grid-template-rows:auto auto;grid-template-columns:auto;position:relative;width:max-content;align-self:end;padding:calc(${designUnit} * 4px) calc(${designUnit} * 4px) 0;box-sizing:border-box}.start,.end{align-self:center}.activeIndicator{grid-row:2;grid-column:1;width:100%;height:5px;justify-self:center;background:${accentFillRest};margin-top:10px;border-radius:calc(${controlCornerRadius} * 1px)
                calc(${controlCornerRadius} * 1px) 0 0}.activeIndicatorTransition{transition:transform 0.2s ease-in-out}.tabpanel{grid-row:2;grid-column-start:1;grid-column-end:4;position:relative}:host([orientation="vertical"]){grid-template-rows:auto 1fr auto;grid-template-columns:auto 1fr}:host([orientation="vertical"]) .tablist{grid-row-start:2;grid-row-end:2;display:grid;grid-template-rows:auto;grid-template-columns:auto 1fr;position:relative;width:max-content;justify-self:end;align-self:flex-start;width:100%;padding:0 calc(${designUnit} * 4px)
                calc((${heightNumber} - ${designUnit}) * 1px) 0}:host([orientation="vertical"]) .tabpanel{grid-column:2;grid-row-start:1;grid-row-end:4}:host([orientation="vertical"]) .end{grid-row:3}:host([orientation="vertical"]) .activeIndicator{grid-column:1;grid-row:1;width:5px;height:100%;margin-inline-end:10px;align-self:center;background:${accentFillRest};margin-top:0;border-radius:0 calc(${controlCornerRadius} * 1px)
                calc(${controlCornerRadius} * 1px) 0}:host([orientation="vertical"]) .activeIndicatorTransition{transition:transform 0.2s linear}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                .activeIndicator,:host([orientation="vertical"]) .activeIndicator{forced-color-adjust:none;background:${SystemColors.Highlight}}`));

/**
 * Styles for Tab
 * @public
 */

const tabStyles = (context, definition) => css`
    ${display("inline-flex")} :host{box-sizing:border-box;font-family:${bodyFont};font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};height:calc(${heightNumber} * 1px);padding:calc(${designUnit} * 5px) calc(${designUnit} * 4px);color:${neutralForegroundHint};fill:currentcolor;border-radius:calc(${controlCornerRadius} * 1px);border:calc(${strokeWidth} * 1px) solid transparent;align-items:center;justify-content:center;grid-row:1;cursor:pointer}:host(:hover){color:${neutralForegroundRest};fill:currentcolor}:host(:active){color:${neutralForegroundRest};fill:currentcolor}:host([disabled]){cursor:${disabledCursor};opacity:${disabledOpacity}}:host([disabled]:hover){color:${neutralForegroundHint};background:${neutralFillStealthRest}}:host([aria-selected="true"]){background:${neutralFillRest};color:${accentForegroundRest};fill:currentcolor}:host([aria-selected="true"]:hover){background:${neutralFillHover};color:${accentForegroundHover};fill:currentcolor}:host([aria-selected="true"]:active){background:${neutralFillActive};color:${accentForegroundActive};fill:currentcolor}:host(:${focusVisible}){outline:none;border:calc(${strokeWidth} * 1px) solid ${focusStrokeOuter};box-shadow:0 0 0 calc((${focusStrokeWidth} - ${strokeWidth}) * 1px)
            ${focusStrokeOuter}}:host(:focus){outline:none}:host(.vertical){justify-content:end;grid-column:2}:host(.vertical[aria-selected="true"]){z-index:2}:host(.vertical:hover){color:${neutralForegroundRest}}:host(.vertical:active){color:${neutralForegroundRest}}:host(.vertical:hover[aria-selected="true"]){}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host{forced-color-adjust:none;border-color:transparent;color:${SystemColors.ButtonText};fill:currentcolor}:host(:hover),:host(.vertical:hover),:host([aria-selected="true"]:hover){background:${SystemColors.Highlight};color:${SystemColors.HighlightText};fill:currentcolor}:host([aria-selected="true"]){background:${SystemColors.HighlightText};color:${SystemColors.Highlight};fill:currentcolor}:host(:${focusVisible}){border-color:${SystemColors.ButtonText};box-shadow:none}:host([disabled]),:host([disabled]:hover){opacity:1;color:${SystemColors.GrayText};background:${SystemColors.ButtonFace}}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#Tab} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#tabTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-tab>`
 */

const fastTab = Tab.compose({
  baseName: "tab",
  template: tabTemplate,
  styles: tabStyles
});

/**
 * Styles for Tab Panel
 * @public
 */

const tabPanelStyles = (context, definition) => css`
    ${display("block")} :host{box-sizing:border-box;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};padding:0 calc((6 + (${designUnit} * 2 * ${density})) * 1px)}`;

/**
 * A function that returns a {@link @microsoft/fast-foundation#TabPanel} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#tabPanelTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-tab-panel>`
 */

const fastTabPanel = TabPanel.compose({
  baseName: "tab-panel",
  template: tabPanelTemplate,
  styles: tabPanelStyles
});

/**
 * A function that returns a {@link @microsoft/fast-foundation#Tabs} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#tabsTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-tabs>`
 */

const fastTabs = Tabs.compose({
  baseName: "tabs",
  template: tabsTemplate,
  styles: tabsStyles
});

/**
 * Styles for Text Area
 * @public
 */

const textAreaStyles = (context, definition) => css`
    ${display("inline-block")} :host{font-family:${bodyFont};outline:none;user-select:none}.control{box-sizing:border-box;position:relative;color:${neutralForegroundRest};background:${neutralFillInputRest};border-radius:calc(${controlCornerRadius} * 1px);border:calc(${strokeWidth} * 1px) solid ${accentFillRest};height:calc(${heightNumber} * 2px);font:inherit;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};padding:calc(${designUnit} * 2px + 1px);width:100%;resize:none}.control:hover:enabled{background:${neutralFillInputHover};border-color:${accentFillHover}}.control:active:enabled{background:${neutralFillInputActive};border-color:${accentFillActive}}.control:hover,.control:${focusVisible},.control:disabled,.control:active{outline:none}:host(:focus-within) .control{border-color:${focusStrokeOuter};box-shadow:0 0 0 1px ${focusStrokeOuter} inset}:host([appearance="filled"]) .control{background:${neutralFillRest}}:host([appearance="filled"]:hover:not([disabled])) .control{background:${neutralFillHover}}:host([resize="both"]) .control{resize:both}:host([resize="horizontal"]) .control{resize:horizontal}:host([resize="vertical"]) .control{resize:vertical}.label{display:block;color:${neutralForegroundRest};cursor:pointer;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};margin-bottom:4px}.label__hidden{display:none;visibility:hidden}:host([disabled]) .label,:host([readonly]) .label,:host([readonly]) .control,:host([disabled]) .control{cursor:${disabledCursor}}:host([disabled]){opacity:${disabledOpacity}}:host([disabled]) .control{border-color:${neutralStrokeRest}}:host([cols]){width:initial}:host([rows]) .control{height:initial}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host([disabled]){opacity:1}`));

/**
 * @internal
 */

class TextArea extends TextArea$1 {
  constructor() {
    super(...arguments);
    /**
     * The appearance of the element.
     *
     * @public
     * @remarks
     * HTML Attribute: appearance
     */

    this.appearance = "outline";
  }

}

__decorate([attr], TextArea.prototype, "appearance", void 0);
/**
 * A function that returns a {@link @microsoft/fast-foundation#TextArea} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#textAreaTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-text-area>`
 *
 * {@link https://developer.mozilla.org/en-US/docs/Web/API/ShadowRoot/delegatesFocus | delegatesFocus}
 */


const fastTextArea = TextArea.compose({
  baseName: "text-area",
  baseClass: TextArea$1,
  template: textAreaTemplate,
  styles: textAreaStyles,
  shadowOptions: {
    delegatesFocus: true
  }
});

/**
 * Styles for Text Field
 * @public
 */

const textFieldStyles = (context, definition) => css`
    ${display("inline-block")} :host{font-family:${bodyFont};outline:none;user-select:none}.root{box-sizing:border-box;position:relative;display:flex;flex-direction:row;color:${neutralForegroundRest};background:${neutralFillInputRest};border-radius:calc(${controlCornerRadius} * 1px);border:calc(${strokeWidth} * 1px) solid ${accentFillRest};height:calc(${heightNumber} * 1px);align-items:baseline}.control{-webkit-appearance:none;font:inherit;background:transparent;border:0;color:inherit;height:calc(100% - 4px);width:100%;margin-top:auto;margin-bottom:auto;border:none;padding:0 calc(${designUnit} * 2px + 1px);font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight}}.control:hover,.control:${focusVisible},.control:disabled,.control:active{outline:none}.label{display:block;color:${neutralForegroundRest};cursor:pointer;font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};margin-bottom:4px}.label__hidden{display:none;visibility:hidden}.start,.control,.end{align-self:center}.start,.end{display:flex;margin:auto;fill:currentcolor}::slotted(svg){width:16px;height:16px}.start{margin-inline-start:11px}.end{margin-inline-end:11px}:host(:hover:not([disabled])) .root{background:${neutralFillInputHover};border-color:${accentFillHover}}:host(:active:not([disabled])) .root{background:${neutralFillInputHover};border-color:${accentFillActive}}:host(:focus-within:not([disabled])) .root{border-color:${focusStrokeOuter};box-shadow:0 0 0 calc(${focusStrokeWidth} * 1px) ${focusStrokeOuter} inset}:host([appearance="filled"]) .root{background:${neutralFillRest}}:host([appearance="filled"]:hover:not([disabled])) .root{background:${neutralFillHover}}:host([disabled]) .label,:host([readonly]) .label,:host([readonly]) .control,:host([disabled]) .control{cursor:${disabledCursor}}:host([disabled]){opacity:${disabledOpacity}}:host([disabled]) .control{border-color:${neutralStrokeRest}}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                .root,:host([appearance="filled"]) .root{forced-color-adjust:none;background:${SystemColors.Field};border-color:${SystemColors.FieldText}}:host(:hover:not([disabled])) .root,:host([appearance="filled"]:hover:not([disabled])) .root,:host([appearance="filled"]:hover) .root{background:${SystemColors.Field};border-color:${SystemColors.Highlight}}.start,.end{fill:currentcolor}:host([disabled]){opacity:1}:host([disabled]) .root,:host([appearance="filled"]:hover[disabled]) .root{border-color:${SystemColors.GrayText};background:${SystemColors.Field}}:host(:focus-within:enabled) .root{border-color:${SystemColors.Highlight};box-shadow:0 0 0 1px ${SystemColors.Highlight} inset}input::placeholder{color:${SystemColors.GrayText}}`));

/**
 * @internal
 */

class TextField extends TextField$1 {
  constructor() {
    super(...arguments);
    /**
     * The appearance of the element.
     *
     * @public
     * @remarks
     * HTML Attribute: appearance
     */

    this.appearance = "outline";
  }

}

__decorate([attr], TextField.prototype, "appearance", void 0);
/**
 * A function that returns a {@link @microsoft/fast-foundation#TextField} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#textFieldTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-text-field>`
 *
 * {@link https://developer.mozilla.org/en-US/docs/Web/API/ShadowRoot/delegatesFocus | delegatesFocus}
 */


const fastTextField = TextField.compose({
  baseName: "text-field",
  baseClass: TextField$1,
  template: textFieldTemplate,
  styles: textFieldStyles,
  shadowOptions: {
    delegatesFocus: true
  }
});

/**
 * Styles for the Toolbar
 *
 * @public
 */

const toolbarStyles = (context, definition) => css`
        ${display("inline-flex")} :host{--toolbar-item-gap:calc(
                (var(--design-unit) + calc(var(--density) + 2)) * 1px
            );background-color:${fillColor};border-radius:calc(${controlCornerRadius} * 1px);fill:currentcolor;padding:var(--toolbar-item-gap)}:host(${focusVisible}){outline:calc(${strokeWidth} * 1px) solid ${neutralStrokeFocus}}.positioning-region{align-items:flex-start;display:inline-flex;flex-flow:row wrap;justify-content:flex-start}:host([orientation="vertical"]) .positioning-region{flex-direction:column}::slotted(:not([slot])){flex:0 0 auto;margin:0 var(--toolbar-item-gap)}:host([orientation="vertical"]) ::slotted(:not([slot])){margin:var(--toolbar-item-gap) 0}.start,.end{display:flex;margin:auto;margin-inline:0}::slotted(svg){width:16px;height:16px}`.withBehaviors(forcedColorsStylesheetBehavior(css`
            :host(:${focusVisible}){box-shadow:0 0 0 calc(${focusStrokeWidth} * 1px) ${SystemColors.Highlight};color:${SystemColors.ButtonText};forced-color-adjust:none}`));

/**
 * @internal
 */

class Toolbar extends Toolbar$1 {
  connectedCallback() {
    super.connectedCallback();
    const parent = composedParent(this);

    if (parent) {
      fillColor.setValueFor(this, target => neutralFillLayerRecipe.getValueFor(target).evaluate(target, fillColor.getValueFor(parent)));
    }
  }

}
/**
 * A function that returns a {@link @microsoft/fast-foundation#Toolbar} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#toolbarTemplate}
 *
 * @public
 * @remarks
 *
 * Generates HTML Element: `<fast-toolbar>`
 *
 */

const fastToolbar = Toolbar.compose({
  baseName: "toolbar",
  baseClass: Toolbar$1,
  template: toolbarTemplate,
  styles: toolbarStyles,
  shadowOptions: {
    delegatesFocus: true
  }
});

/**
 * Styles for Tooltip
 * @public
 */

const tooltipStyles = (context, definition) => {
  const anchoredRegionTag = context.tagFor(AnchoredRegion);
  return css`
            :host{contain:size;overflow:visible;height:0;width:0}.tooltip{box-sizing:border-box;border-radius:calc(${controlCornerRadius} * 1px);border:calc(${strokeWidth} * 1px) solid ${focusStrokeOuter};box-shadow:0 0 0 1px ${focusStrokeOuter} inset;background:${neutralFillRest};color:${neutralForegroundRest};padding:4px;height:fit-content;width:fit-content;font-family:${bodyFont};font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};white-space:nowrap;z-index:10000}${anchoredRegionTag}{display:flex;justify-content:center;align-items:center;overflow:visible;flex-direction:row}${anchoredRegionTag}.right,${anchoredRegionTag}.left{flex-direction:column}${anchoredRegionTag}.top .tooltip{margin-bottom:4px}${anchoredRegionTag}.bottom .tooltip{margin-top:4px}${anchoredRegionTag}.left .tooltip{margin-right:4px}${anchoredRegionTag}.right .tooltip{margin-left:4px}${anchoredRegionTag}.top.left .tooltip,${anchoredRegionTag}.top.right .tooltip{margin-bottom:0px}${anchoredRegionTag}.bottom.left .tooltip,${anchoredRegionTag}.bottom.right .tooltip{margin-top:0px}${anchoredRegionTag}.top.left .tooltip,${anchoredRegionTag}.bottom.left .tooltip{margin-right:0px}${anchoredRegionTag}.top.right .tooltip,${anchoredRegionTag}.bottom.right .tooltip{margin-left:0px}`.withBehaviors(forcedColorsStylesheetBehavior(css`
                :host([disabled]){opacity:1}`));
};

/**
 * A function that returns a {@link @microsoft/fast-foundation#Tooltip} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#tooltipTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-tooltip>`
 */

const fastTooltip = Tooltip.compose({
  baseName: "tooltip",
  template: tooltipTemplate,
  styles: tooltipStyles
});

const ltr = css`
    .expand-collapse-glyph{transform:rotate(0deg)}:host(.nested) .expand-collapse-button{left:var(--expand-collapse-button-nested-width,calc(${heightNumber} * -1px))}:host([selected])::after{left:calc(${focusStrokeWidth} * 1px)}:host([expanded]) > .positioning-region .expand-collapse-glyph{transform:rotate(45deg)}`;
const rtl = css`
    .expand-collapse-glyph{transform:rotate(180deg)}:host(.nested) .expand-collapse-button{right:var(--expand-collapse-button-nested-width,calc(${heightNumber} * -1px))}:host([selected])::after{right:calc(${focusStrokeWidth} * 1px)}:host([expanded]) > .positioning-region .expand-collapse-glyph{transform:rotate(135deg)}`;
/**
 * Tree item expand collapse button size CSS Partial
 * @public
 */

const expandCollapseButtonSize = cssPartial`((${baseHeightMultiplier} / 2) * ${designUnit}) + ((${designUnit} * ${density}) / 2)`;
const expandCollapseHoverBehavior = DesignToken.create("tree-item-expand-collapse-hover").withDefault(target => {
  const recipe = neutralFillStealthRecipe.getValueFor(target);
  return recipe.evaluate(target, recipe.evaluate(target).hover).hover;
});
const selectedExpandCollapseHoverBehavior = DesignToken.create("tree-item-expand-collapse-selected-hover").withDefault(target => {
  const baseRecipe = neutralFillRecipe.getValueFor(target);
  const buttonRecipe = neutralFillStealthRecipe.getValueFor(target);
  return buttonRecipe.evaluate(target, baseRecipe.evaluate(target).rest).hover;
});
/**
 * Styles for Tree Item
 * @public
 */

const treeItemStyles = (context, definition) => css`@keyframes treeItemLoading{0%{opacity:0}100%{opacity:1}}${display("block")} :host{contain:content;position:relative;outline:none;color:${neutralForegroundRest};background:${neutralFillStealthRest};cursor:pointer;font-family:${bodyFont};--expand-collapse-button-size:calc(${heightNumber} * 1px);--tree-item-nested-width:0}:host(:focus) > .positioning-region{outline:none}:host(:focus) .content-region{outline:none}:host(:${focusVisible}) .positioning-region{border:${focusStrokeOuter} calc(${strokeWidth} * 1px) solid;border-radius:calc(${controlCornerRadius} * 1px);color:${neutralForegroundRest}}.positioning-region{display:flex;position:relative;box-sizing:border-box;background:${neutralFillStealthRest};border:transparent calc(${strokeWidth} * 1px) solid;height:calc((${heightNumber} + 1) * 1px)}.positioning-region::before{content:"";display:block;width:var(--tree-item-nested-width);flex-shrink:0}:host(:not([disabled])) .positioning-region:hover{background:${neutralFillStealthHover}}:host(:not([disabled])) .positioning-region:active{background:${neutralFillStealthActive}}.content-region{display:inline-flex;align-items:center;white-space:nowrap;width:100%;height:calc(${heightNumber} * 1px);margin-inline-start:calc(${designUnit} * 2px + 8px);font-size:${typeRampBaseFontSize};line-height:${typeRampBaseLineHeight};font-weight:400}.items{font-size:calc(1em + (${designUnit} + 16) * 1px)}.expand-collapse-button{background:none;border:none;outline:none;width:calc((${expandCollapseButtonSize} + (${designUnit} * 2)) * 1px);height:calc((${expandCollapseButtonSize} + (${designUnit} * 2)) * 1px);padding:0;display:flex;justify-content:center;align-items:center;cursor:pointer;margin-left:6px;margin-right:6px}.expand-collapse-glyph{width:16px;height:16px;transition:transform 0.1s linear;pointer-events:none;fill:currentcolor}.start,.end{display:flex;fill:currentcolor}::slotted(svg){width:16px;height:16px}.start{margin-inline-end:calc(${designUnit} * 2px + 2px)}.end{margin-inline-start:calc(${designUnit} * 2px + 2px)}:host([expanded]) > .items{animation:treeItemLoading ease-in 10ms;animation-iteration-count:1;animation-fill-mode:forwards}:host([disabled]) .content-region{opacity:${disabledOpacity};cursor:${disabledCursor}}:host(.nested) .content-region{position:relative;margin-inline-start:var(--expand-collapse-button-size)}:host(.nested) .expand-collapse-button{position:absolute}:host(.nested:not([disabled])) .expand-collapse-button:hover{background:${expandCollapseHoverBehavior}}:host([selected]) .positioning-region{background:${neutralFillRest}}:host([selected]:not([disabled])) .positioning-region:hover{background:${neutralFillHover}}:host([selected]:not([disabled])) .positioning-region:active{background:${neutralFillActive}}:host([selected]:not([disabled])) .expand-collapse-button:hover{background:${selectedExpandCollapseHoverBehavior}}:host([selected])::after{background:${accentForegroundRest};border-radius:calc(${controlCornerRadius} * 1px);content:"";display:block;position:absolute;top:calc((${heightNumber} / 4) * 1px);width:3px;height:calc((${heightNumber} / 2) * 1px)}::slotted(${context.tagFor(TreeItem)}){--tree-item-nested-width:1em;--expand-collapse-button-nested-width:calc(${heightNumber} * -1px)}`.withBehaviors(new DirectionalStyleSheetBehavior(ltr, rtl), forcedColorsStylesheetBehavior(css`
            :host{forced-color-adjust:none;border-color:transparent;background:${SystemColors.Field};color:${SystemColors.FieldText}}:host .content-region .expand-collapse-glyph{fill:${SystemColors.FieldText}}:host .positioning-region:hover,:host([selected]) .positioning-region{background:${SystemColors.Highlight}}:host .positioning-region:hover .content-region,:host([selected]) .positioning-region .content-region{color:${SystemColors.HighlightText}}:host .positioning-region:hover .content-region .expand-collapse-glyph,:host .positioning-region:hover .content-region .start,:host .positioning-region:hover .content-region .end,:host([selected]) .content-region .expand-collapse-glyph,:host([selected]) .content-region .start,:host([selected]) .content-region .end{fill:${SystemColors.HighlightText}}:host([selected])::after{background:${SystemColors.Field}}:host(:${focusVisible}) .positioning-region{border-color:${SystemColors.FieldText};box-shadow:0 0 0 2px inset ${SystemColors.Field};color:${SystemColors.FieldText}}:host([disabled]) .content-region,:host([disabled]) .positioning-region:hover .content-region{opacity:1;color:${SystemColors.GrayText}}:host([disabled]) .content-region .expand-collapse-glyph,:host([disabled]) .content-region .start,:host([disabled]) .content-region .end,:host([disabled]) .positioning-region:hover .content-region .expand-collapse-glyph,:host([disabled]) .positioning-region:hover .content-region .start,:host([disabled]) .positioning-region:hover .content-region .end{fill:${SystemColors.GrayText}}:host([disabled]) .positioning-region:hover{background:${SystemColors.Field}}.expand-collapse-glyph,.start,.end{fill:${SystemColors.FieldText}}:host(.nested) .expand-collapse-button:hover{background:${SystemColors.Field}}:host(.nested) .expand-collapse-button:hover .expand-collapse-glyph{fill:${SystemColors.FieldText}}`));

/**
 * A function that returns a {@link @microsoft/fast-foundation#TreeItem} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#treeItemTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-tree-item>`
 *
 */

const fastTreeItem = TreeItem.compose({
  baseName: "tree-item",
  template: treeItemTemplate,
  styles: treeItemStyles,
  expandCollapseGlyph:
  /* html */
  `
        <svg
            viewBox="0 0 16 16"
            xmlns="http://www.w3.org/2000/svg"
            class="expand-collapse-glyph"
        >
            <path
                d="M5.00001 12.3263C5.00124 12.5147 5.05566 12.699 5.15699 12.8578C5.25831 13.0167 5.40243 13.1437 5.57273 13.2242C5.74304 13.3047 5.9326 13.3354 6.11959 13.3128C6.30659 13.2902 6.4834 13.2152 6.62967 13.0965L10.8988 8.83532C11.0739 8.69473 11.2153 8.51658 11.3124 8.31402C11.4096 8.11146 11.46 7.88966 11.46 7.66499C11.46 7.44033 11.4096 7.21853 11.3124 7.01597C11.2153 6.81341 11.0739 6.63526 10.8988 6.49467L6.62967 2.22347C6.48274 2.10422 6.30501 2.02912 6.11712 2.00691C5.92923 1.9847 5.73889 2.01628 5.56823 2.09799C5.39757 2.17969 5.25358 2.30817 5.153 2.46849C5.05241 2.62882 4.99936 2.8144 5.00001 3.00369V12.3263Z"
            />
        </svg>
    `
});

/**
 * Styles for Tree View
 * @public
 */

const treeViewStyles = (context, definition) => css`
    ${display("flex")} :host{flex-direction:column;align-items:stretch;min-width:fit-content;font-size:0}:host:focus-visible{outline:none}`;

/**
 * A function that returns a {@link @microsoft/fast-foundation#TreeView} registration for configuring the component with a DesignSystem.
 * Implements {@link @microsoft/fast-foundation#treeViewTemplate}
 *
 *
 * @public
 * @remarks
 * Generates HTML Element: `<fast-tree-view>`
 *
 */

const fastTreeView = TreeView.compose({
  baseName: "tree-view",
  template: treeViewTemplate,
  styles: treeViewStyles
});

/**
 * Export all custom element definitions
 */
/**
 * All Web Components
 * @public
 * @remarks
 * This object can be passed directly to the Design System's `register` method to
 * statically link and register all available components.
 */

const allComponents = {
  fastAccordion,
  fastAccordionItem,
  fastAnchor,
  fastAnchoredRegion,
  fastAvatar,
  fastBadge,
  fastBreadcrumb,
  fastBreadcrumbItem,
  fastButton,
  fastCalendar,
  fastCard,
  fastCheckbox,
  fastCombobox,
  fastDataGrid,
  fastDataGridCell,
  fastDataGridRow,
  fastDesignSystemProvider,
  fastDialog,
  fastDisclosure,
  fastDivider,
  fastFlipper,
  fastHorizontalScroll,
  fastListbox,
  fastOption,
  fastMenu,
  fastMenuItem,
  fastNumberField,
  fastPicker,
  fastPickerList,
  fastPickerListItem,
  fastPickerMenu,
  fastPickerMenuOption,
  fastProgress,
  fastProgressRing,
  fastRadio,
  fastRadioGroup,
  fastSearch,
  fastSelect,
  fastSkeleton,
  fastSlider,
  fastSliderLabel,
  fastSwitch,
  fastTabs,
  fastTab,
  fastTabPanel,
  fastTextArea,
  fastTextField,
  fastTooltip,
  fastToolbar,
  fastTreeView,
  fastTreeItem,

  register(container, ...rest) {
    if (!container) {
      // preserve backward compatibility with code that loops through
      // the values of this object and calls them as funcs with no args
      return;
    }

    for (const key in this) {
      if (key === "register") {
        continue;
      }

      this[key]().register(container, ...rest);
    }
  }

};

/**
 * Provides a design system for the specified element either by returning one that was
 * already created for that element or creating one.
 * @param element - The element to root the design system at. By default, this is the body.
 * @returns A FAST Design System
 * @public
 */

function provideFASTDesignSystem(element) {
  return DesignSystem.getOrCreate(element).withPrefix("fast");
}

// TODO: Is exporting Foundation still necessary with the updated API's?
/**
 * The global FAST Design System.
 * @remarks
 * Only available if the components are added through a script tag
 * rather than a module/build system.
 */

const FASTDesignSystem = provideFASTDesignSystem().register(allComponents);

export { Accordion, AccordionItem, ActionsStyles, Anchor, AnchoredRegion, Avatar, Badge, Breadcrumb, BreadcrumbItem, Button, Calendar, CalendarStyles, Card, Checkbox, Combobox, DataGrid, DataGridCell, DataGridRow, DesignSystemProvider, Dialog, DirectionalStyleSheetBehavior, Disclosure, Divider, FASTDesignSystem, Flipper, HorizontalScroll, Listbox, ListboxOption, Menu, MenuItem, NumberField, PaletteRGB, Picker, PickerMenu, BaseProgress as Progress, BaseProgress as ProgressRing, Radio, RadioGroup, Search, Select, Skeleton, Slider, SliderLabel, StandardLuminance, SwatchRGB, Switch, Tab, TabPanel, Tabs, TextArea, TextField, Toolbar, Tooltip, TreeItem, TreeView, accentColor, accentFillActive, accentFillActiveDelta, accentFillFocus, accentFillFocusDelta, accentFillHover, accentFillHoverDelta, accentFillRecipe, accentFillRest, accentFillRestDelta, accentForegroundActive, accentForegroundActiveDelta, accentForegroundFocus, accentForegroundFocusDelta, accentForegroundHover, accentForegroundHoverDelta, accentForegroundRecipe, accentForegroundRest, accentForegroundRestDelta, accentPalette, accordionItemStyles, accordionStyles, allComponents, anchorStyles, anchoredRegionStyles, avatarStyles, badgeStyles, baseHeightMultiplier, baseHorizontalSpacingMultiplier, baseLayerLuminance, bodyFont, breadcrumbItemStyles, breadcrumbStyles, buttonStyles, cardStyles, checkboxStyles, comboboxStyles, controlCornerRadius, dataGridCellStyles, dataGridRowStyles, dataGridStyles, density, designSystemProviderStyles, designSystemProviderTemplate, designUnit, dialogStyles, direction, disabledOpacity, disclosureStyles, dividerStyles, fastAccordion, fastAccordionItem, fastAnchor, fastAnchoredRegion, fastAvatar, fastBadge, fastBreadcrumb, fastBreadcrumbItem, fastButton, fastCalendar, fastCard, fastCheckbox, fastCombobox, fastDataGrid, fastDataGridCell, fastDataGridRow, fastDesignSystemProvider, fastDialog, fastDisclosure, fastDivider, fastFlipper, fastHorizontalScroll, fastListbox, fastMenu, fastMenuItem, fastNumberField, fastOption, fastPicker, fastPickerList, fastPickerListItem, fastPickerMenu, fastPickerMenuOption, fastProgress, fastProgressRing, fastRadio, fastRadioGroup, fastSearch, fastSelect, fastSkeleton, fastSlider, fastSliderLabel, fastSwitch, fastTab, fastTabPanel, fastTabs, fastTextArea, fastTextField, fastToolbar, fastTooltip, fastTreeItem, fastTreeView, fillColor, flipperStyles, focusStrokeInner, focusStrokeInnerRecipe, focusStrokeOuter, focusStrokeOuterRecipe, focusStrokeWidth, foregroundOnAccentActive, foregroundOnAccentActiveLarge, foregroundOnAccentFocus, foregroundOnAccentFocusLarge, foregroundOnAccentHover, foregroundOnAccentHoverLarge, foregroundOnAccentLargeRecipe, foregroundOnAccentRecipe, foregroundOnAccentRest, foregroundOnAccentRestLarge, heightNumberAsToken, horizontalScrollStyles, horizontalSliderLabelStyles, imgTemplate, isDark, listboxStyles, menuItemStyles, menuStyles, neutralColor, neutralFillActive, neutralFillActiveDelta, neutralFillFocus, neutralFillFocusDelta, neutralFillHover, neutralFillHoverDelta, neutralFillInputActive, neutralFillInputActiveDelta, neutralFillInputFocus, neutralFillInputFocusDelta, neutralFillInputHover, neutralFillInputHoverDelta, neutralFillInputRecipe, neutralFillInputRest, neutralFillInputRestDelta, neutralFillLayerRecipe, neutralFillLayerRest, neutralFillLayerRestDelta, neutralFillRecipe, neutralFillRest, neutralFillRestDelta, neutralFillStealthActive, neutralFillStealthActiveDelta, neutralFillStealthFocus, neutralFillStealthFocusDelta, neutralFillStealthHover, neutralFillStealthHoverDelta, neutralFillStealthRecipe, neutralFillStealthRest, neutralFillStealthRestDelta, neutralFillStrongActive, neutralFillStrongActiveDelta, neutralFillStrongFocus, neutralFillStrongFocusDelta, neutralFillStrongHover, neutralFillStrongHoverDelta, neutralFillStrongRecipe, neutralFillStrongRest, neutralFillStrongRestDelta, neutralForegroundHint, neutralForegroundHintRecipe, neutralForegroundRecipe, neutralForegroundRest, neutralLayer1, neutralLayer1Recipe, neutralLayer2, neutralLayer2Recipe, neutralLayer3, neutralLayer3Recipe, neutralLayer4, neutralLayer4Recipe, neutralLayerCardContainer, neutralLayerCardContainerRecipe, neutralLayerFloating, neutralLayerFloatingRecipe, neutralPalette, neutralStrokeActive, neutralStrokeActiveDelta, neutralStrokeDividerRecipe, neutralStrokeDividerRest, neutralStrokeDividerRestDelta, neutralStrokeFocus, neutralStrokeFocusDelta, neutralStrokeHover, neutralStrokeHoverDelta, neutralStrokeRecipe, neutralStrokeRest, neutralStrokeRestDelta, numberFieldStyles, optionStyles, pickerListItemStyles, pickerMenuOptionStyles, pickerMenuStyles, pickerStyles, progressRingStyles, progressStyles, provideFASTDesignSystem, radioGroupStyles, radioStyles, searchStyles, selectStyles, skeletonStyles, sliderLabelStyles, sliderStyles, strokeWidth, switchStyles, tabPanelStyles, tabStyles, tabsStyles, textAreaStyles, textFieldStyles, toolbarStyles, tooltipStyles, treeItemStyles, treeViewStyles, typeRampBaseFontSize, typeRampBaseLineHeight, typeRampMinus1FontSize, typeRampMinus1LineHeight, typeRampMinus2FontSize, typeRampMinus2LineHeight, typeRampPlus1FontSize, typeRampPlus1LineHeight, typeRampPlus2FontSize, typeRampPlus2LineHeight, typeRampPlus3FontSize, typeRampPlus3LineHeight, typeRampPlus4FontSize, typeRampPlus4LineHeight, typeRampPlus5FontSize, typeRampPlus5LineHeight, typeRampPlus6FontSize, typeRampPlus6LineHeight, verticalSliderLabelStyles };