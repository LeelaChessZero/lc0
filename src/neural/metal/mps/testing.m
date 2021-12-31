//
//  testing.m
//  lc0mps
//
//  Created by KodeHauz Mac1 on 30/12/2021.
//

#import <Foundation/Foundation.h>

int main(int argc, const char * argv[]) {
    NSArray *allPaths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [allPaths objectAtIndex:0];
    NSString *pathForLog = [documentsDirectory stringByAppendingPathComponent:@"lc0logfile.txt"];
    
    freopen([pathForLog cStringUsingEncoding:NSASCIIStringEncoding], "a+", stderr);
    
    NSLog(@"Testing");
    
    return 0;
}
