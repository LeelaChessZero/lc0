#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "sysport.h"

/**** CLOCK *************************************************************************/

#if defined(USECLOCK)

	#include <time.h>
	extern myclock_t myclock(void) {return (myclock_t)clock();}
	extern myclock_t ticks_per_sec (void) {return CLOCKS_PER_SEC;}

#elif defined(USEWINCLOCK)

	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
	extern myclock_t myclock(void) {return (myclock_t)GetTickCount();}
	extern myclock_t ticks_per_sec (void) {return 1000;}


#elif defined(USELINCLOCK)

	#include <sys/time.h>
	extern myclock_t myclock(void) 
	{
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return (myclock_t)tv.tv_sec * 1000 + (myclock_t)tv.tv_usec/1000;
	}
	extern myclock_t ticks_per_sec (void) {return 1000;}

#else

	#error No Clock specified in compilation

#endif

/**** PATH NAMES *************************************************************************/

#if defined(GCCLINUX)
	extern int isfoldersep (int x) { return x == '/';}
#elif defined(MVSC)
	extern int isfoldersep (int x) { return x == '\\' || x == ':';}
#else
	extern int isfoldersep (int x) { return x == '/' || x == '\\' || x == ':';}
#endif

/**** Maximum Files Open *****************************************************************/

#if defined(GCCLINUX)
	#include <sys/resource.h>
	#if 0	
	struct rlimit {
		rlim_t rlim_cur;  /* Soft limit */
		rlim_t rlim_max;  /* Hard limit (ceiling for rlim_cur) */
	};
	#endif
	extern int mysys_fopen_max (void) 
	{ 
		int ok;
		struct rlimit rl;
		ok = 0 == getrlimit(RLIMIT_NOFILE, &rl);
		if (ok)
			return (int)rl.rlim_cur;
		else
			return FOPEN_MAX;
	}
#elif defined(MVSC)
	extern int mysys_fopen_max (void) { return FOPEN_MAX;}
#else
	extern int mysys_fopen_max (void) { return FOPEN_MAX;}
#endif


#if defined(MULTI_THREADED_INTERFACE)
/**** THREADS ****************************************************************************/

/*
|
|	POSIX
|
\*-------------------------*/
#if defined (POSIX_THREADS)

#include <pthread.h>

extern int /* boolean */
mythread_create (/*@out@*/ mythread_t *thread, routine_t start_routine, void *arg, /*@out@*/ int *ret_error)
{
	const pthread_attr_t *attr = NULL; /* default attributes */
	int ret;
	ret = pthread_create (thread, attr, start_routine, arg);
	*ret_error = ret;
	return 0 == ret;
}

extern int /* boolean */
mythread_join (mythread_t thread)
{
	void *p; /* value return from pthread_exit, not used */
	int ret = pthread_join (thread, &p);
	return 0 == ret;
}

extern void 		
mythread_exit (void)
{
	pthread_exit (NULL);
}


extern const char *
mythread_create_error (int err)
{
	const char *s;
	switch (err) {
		case 0     : s = "Success"; break;
		case EAGAIN: s = "EAGAIN" ; break;
		case EINVAL: s = "EINVAL" ; break; 
		case EPERM : s = "EPERM"  ; break;
		default    : s = "Unknown error"; break;
	}
	return s;
}

extern void mythread_mutex_init		(mythread_mutex_t *m) { pthread_mutex_init   (m,NULL);}
extern void mythread_mutex_destroy	(mythread_mutex_t *m) { pthread_mutex_destroy(m)     ;}
extern void mythread_mutex_lock     (mythread_mutex_t *m) { pthread_mutex_lock   (m)     ;}
extern void mythread_mutex_unlock   (mythread_mutex_t *m) { pthread_mutex_unlock (m)     ;}

#ifdef SPINLOCKS
extern void mythread_spinx_init		(mythread_spinx_t *m) { pthread_spin_init   (m,0);} /**/
extern void mythread_spinx_destroy	(mythread_spinx_t *m) { pthread_spin_destroy(m)  ;} /**/
extern void mythread_spinx_lock     (mythread_spinx_t *m) { pthread_spin_lock   (m)  ;} /**/
extern void mythread_spinx_unlock   (mythread_spinx_t *m) { pthread_spin_unlock (m)  ;} /**/
#else
extern void mythread_spinx_init		(mythread_spinx_t *m) { pthread_mutex_init   (m,NULL);} /**/
extern void mythread_spinx_destroy	(mythread_spinx_t *m) { pthread_mutex_destroy(m)     ;} /**/
extern void mythread_spinx_lock     (mythread_spinx_t *m) { pthread_mutex_lock   (m)     ;} /**/
extern void mythread_spinx_unlock   (mythread_spinx_t *m) { pthread_mutex_unlock (m)     ;} /**/
#endif

/* semaphores */
extern int /* boolean */ 
mysem_init	(mysem_t *sem, unsigned int value)
	{ return -1 != sem_init (sem, 0 /*not shared with processes*/, value);}

extern int /* boolean */ 
mysem_wait	(mysem_t *sem)
	{ return  0 == sem_wait (sem);}

extern int /* boolean */ 
mysem_post	(mysem_t *sem)
	{ return  0 == sem_post (sem);}
 
extern int /* boolean */ 
mysem_destroy	(mysem_t *sem)
	{ return  0 == sem_destroy (sem);}

/*
|
|	NT_THREADS
|
\*-------------------------*/
#elif defined(NT_THREADS)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>

extern int /* boolean */
mythread_create (/*@out@*/ mythread_t *thread, routine_t start_routine, void *arg, /*@out@*/ int *ret_error)
{
	static unsigned int	thread_id;
	mythread_t t;
	int /* boolean */ is_ok;

	t =	(mythread_t) _beginthreadex (NULL, 0, start_routine, arg, 0, &thread_id );
	is_ok = (t != 0);
	*thread = t;
	*ret_error = is_ok? 0: errno;
	return is_ok;
}

extern int /* boolean */
mythread_join (mythread_t thread)
{
	unsigned long int ret;
	ret = WaitForSingleObject (thread, INFINITE);
	CloseHandle(thread);
	return ret != WAIT_FAILED;
}

extern void 		
mythread_exit (void)
{
	return;
}

extern const char *
mythread_create_error (int err)
{
	const char *s;
	switch (err) {
		case 0     : s = "Success"; break;
		case EAGAIN: s = "EAGAIN" ; break;
		case EINVAL: s = "EINVAL" ; break; 
		case EPERM : s = "EPERM"  ; break;
		default    : s = "Unknown error"; break;
	}
	return s;
}

extern void mythread_mutex_init		(mythread_mutex_t *m) { *m = CreateMutex(0, FALSE, 0)      ;}
extern void mythread_mutex_destroy	(mythread_mutex_t *m) { CloseHandle(*m)                    ;}
extern void mythread_mutex_lock     (mythread_mutex_t *m) { WaitForSingleObject(*m, INFINITE)  ;}
extern void mythread_mutex_unlock   (mythread_mutex_t *m) { ReleaseMutex(*m)                   ;}

extern void mythread_spinx_init		(mythread_spinx_t *m) { InitializeCriticalSection(m)  ;} /**/
extern void mythread_spinx_destroy	(mythread_spinx_t *m) { DeleteCriticalSection(m)  ;} /**/
extern void mythread_spinx_lock     (mythread_spinx_t *m) { EnterCriticalSection (m)  ;} /**/
extern void mythread_spinx_unlock   (mythread_spinx_t *m) { LeaveCriticalSection (m)  ;} /**/

/* semaphores */
extern int /* boolean */ 
mysem_init	(mysem_t *sem, unsigned int value)
{
	mysem_t h =
	CreateSemaphore(
  		NULL, 		/* cannot be inherited */
		(LONG)value,/* Initial Count */
  		256,		/* Maximum Count */
  		NULL		/* Name --> NULL, not shared among threads */
	);

	if (h != NULL)	*sem = h;

	return h != NULL;
}

extern int /* boolean */ 
mysem_wait	(mysem_t *sem)
{ 
	HANDLE h = *sem;
	return WAIT_FAILED != WaitForSingleObject (h, INFINITE); 
}

extern int /* boolean */ 
mysem_post	(mysem_t *sem)
{ 
	HANDLE h = *sem;
	return 0 != ReleaseSemaphore(h, 1, NULL);	
}

extern int /* boolean */ 
mysem_destroy	(mysem_t *sem)
{ 
	return 0 != CloseHandle( *sem);
}

/**** THREADS ****************************************************************************/
#else
	#error Definition of threads not present
#endif

/* MULTI_THREADED_INTERFACE */
#endif






